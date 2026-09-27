# SOMA Phase 8 — C ABI, Fortran, Python, LowFive, Demo B

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §3.2 (narrow waist), §6 (all four integration modes), §11 Demo B
**Milestone:** two **unmodified** HDF5-writing executables are coupled purely by intercepting their I/O, through the same runtime that drives the in-process C++ components of Demo A.

This phase cashes the architectural cheque from spec §2.2: `Transport` is one seam with four implementations, and non-intrusive coupling is not a parallel code path.

---

## Environment blocker, verified 2026-09-02

**LowFive is installed but cannot load.**

```
dlopen(/opt/lowfive/lib/liblowfive.dylib):
  Library not loaded: /opt/hdf5/lib/libhdf5.310.dylib
```

`/opt/hdf5` contains only static archives — `libhdf5.a`, `libhdf5_hl.a`, no
`.dylib`. A runtime-loaded HDF5 VOL plugin cannot be backed by a static HDF5, so
Task 8.0 rebuilds HDF5 as shared. Also noted: `/opt/lowfive/lib/python3.11`
targets Python 3.11 (`/opt/homebrew/bin/python3.11`), while the system default
is 3.13 — Task 8.3 pins the interpreter accordingly.

Tasks 8.1, 8.2, and 8.3 do not depend on LowFive and proceed regardless.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/soma_c.h` | the C ABI — the narrow waist |
| `src/capi/soma_c.cpp` | C ABI implementation |
| `fortran/soma_mod.f90` | `ISO_C_BINDING` module |
| `python/soma/__init__.py`, `python/src/bindings.cpp` | pybind11 module |
| `src/transport/lowfive_transport.cpp` | HDF5-VOL interception |
| `examples/lowfive_demo/` | Demo B |

---

### Task 8.0: Rebuild HDF5 as shared and revalidate LowFive

**Files:**
- Modify: `docs/BUILDING.md`
- Create: `cmake/FindSomaLowFive.cmake`

- [ ] **Step 1: Record the failure before changing anything**

```bash
ls /opt/hdf5/lib/
grep -E "HDF5 Version|Parallel HDF5|Shared Library" /opt/hdf5/lib/libhdf5.settings
otool -L /opt/lowfive/lib/liblowfive.dylib | grep -i hdf5
cat > /tmp/lftest.c <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int main(void){ void*h=dlopen("/opt/lowfive/lib/liblowfive.dylib",RTLD_NOW);
  printf(h?"OK\n":"FAIL: %s\n", h?"":dlerror()); return h?0:1; }
EOF
cc /tmp/lftest.c -o /tmp/lftest && /tmp/lftest
```

Expected: `FAIL: ... libhdf5.310.dylib ... no such file`. Paste this verbatim
into `docs/BUILDING.md` as the "before" record — this is the class of
environment problem the KARMA handoff shows costs days when undocumented.

- [ ] **Step 2: Build HDF5 1.14.1 as shared, parallel, into a new prefix**

Do not overwrite `/opt/hdf5`: MOAB and PETSc link against it.

```bash
export SOMA_HDF5_PREFIX=/opt/hdf5-shared
curl -L -o /tmp/hdf5.tar.gz \
  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.1-2/src/hdf5-1.14.1-2.tar.gz
tar xf /tmp/hdf5.tar.gz -C /tmp && cd /tmp/hdf5-1.14.1-2
CC=/opt/mpich/bin/mpicc ./configure --prefix=$SOMA_HDF5_PREFIX \
    --enable-shared --disable-static --enable-parallel --enable-hl
make -j8 && make install
ls $SOMA_HDF5_PREFIX/lib/libhdf5*.dylib
```

Version 1.14.1 is chosen to match what LowFive was built against
(`libhdf5.310.dylib` is 1.14.1's soname); a different major version would need
LowFive rebuilt too.

- [ ] **Step 3: Rebuild LowFive against the shared HDF5**

```bash
export SOMA_LOWFIVE_PREFIX=/opt/lowfive-shared
git clone --recursive https://github.com/diatomic/LowFive.git /tmp/lowfive-src
cmake -B /tmp/lowfive-build -S /tmp/lowfive-src \
      -DCMAKE_INSTALL_PREFIX=$SOMA_LOWFIVE_PREFIX \
      -DCMAKE_PREFIX_PATH="$SOMA_HDF5_PREFIX;/opt/diy" \
      -DCMAKE_CXX_COMPILER=/opt/mpich/bin/mpicxx \
      -DBUILD_SHARED_LIBS=ON \
      -Dbuild_examples=OFF
cmake --build /tmp/lowfive-build -j8 && cmake --install /tmp/lowfive-build
```

- [ ] **Step 4: Verify it loads**

```bash
sed -i.bak "s|/opt/lowfive|$SOMA_LOWFIVE_PREFIX|" /tmp/lftest.c
cc /tmp/lftest.c -o /tmp/lftest && /tmp/lftest
otool -L $SOMA_LOWFIVE_PREFIX/lib/liblowfive.dylib | grep -i hdf5
```

Expected: `OK`, and the linkage pointing at `/opt/hdf5-shared`.

**If this still fails, stop.** Record the outcome in `docs/BUILDING.md`, mark
Tasks 8.4 and 8.5 blocked, and complete 8.1–8.3, which have no LowFive
dependency. Do not fake Demo B with a file-copy stand-in — it would test nothing
and read as a passing feature.

- [ ] **Step 5: Write `cmake/FindSomaLowFive.cmake`**

```cmake
# LowFive is a runtime-loaded HDF5 VOL plugin, so both the library and a
# SHARED HDF5 must be present. A static HDF5 cannot back a VOL plugin.

find_path(LOWFIVE_INCLUDE_DIR lowfive/vol-base.hpp
          HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES include)
find_library(LOWFIVE_LIBRARY NAMES lowfive
             HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_package(HDF5 COMPONENTS C HL)

if(NOT LOWFIVE_INCLUDE_DIR OR NOT LOWFIVE_LIBRARY)
  set(SOMA_HAVE_LOWFIVE OFF PARENT_SCOPE)
  message(STATUS "LowFive: not found; Demo B is unavailable")
  return()
endif()

# A .a here means the plugin cannot be dlopened -- catch it at configure time
# rather than at run time.
get_filename_component(_lf_ext "${LOWFIVE_LIBRARY}" EXT)
if(_lf_ext STREQUAL ".a")
  set(SOMA_HAVE_LOWFIVE OFF PARENT_SCOPE)
  message(WARNING
    "LowFive: found a static archive (${LOWFIVE_LIBRARY}). A VOL plugin must "
    "be shared. See docs/BUILDING.md.")
  return()
endif()

if(HDF5_FOUND AND NOT HDF5_IS_PARALLEL)
  message(WARNING "LowFive: HDF5 is not parallel; n->m redistribution will fail")
endif()

set(SOMA_HAVE_LOWFIVE ON PARENT_SCOPE)
add_library(soma::lowfive UNKNOWN IMPORTED)
set_target_properties(soma::lowfive PROPERTIES
  IMPORTED_LOCATION "${LOWFIVE_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${LOWFIVE_INCLUDE_DIR}")
```

Wire it into the top-level `CMakeLists.txt` under `if(SOMA_ENABLE_LOWFIVE)`.

- [ ] **Step 6: Commit**

```bash
git add cmake/FindSomaLowFive.cmake CMakeLists.txt docs/BUILDING.md
git commit -m "build(soma): LowFive discovery with shared-HDF5 validation"
```

---

### Task 8.1: The C ABI

Spec §3.2: one entry path shared by C++, Fortran, Python, and YAML. Everything
crosses as the serialized graph descriptor from Phase 1 Task 1.6.

**Files:**
- Create: `include/soma/soma_c.h`
- Create: `src/capi/soma_c.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/capi_test.c`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Produces, all `extern "C"`, all returning `int` (0 = success, negative = error code):
  - opaque `typedef struct soma_ctx_s* soma_ctx;`
  - `int soma_create(soma_ctx* out)`, `int soma_destroy(soma_ctx)`
  - `int soma_load_graph(soma_ctx, const void* descriptor, size_t bytes)` — the narrow waist
  - `int soma_load_graph_yaml(soma_ctx, const char* path)`
  - `int soma_field_handle(soma_ctx, const char* component, const char* field, int* handle)`
  - `int soma_register_field(soma_ctx, int handle, double* data, size_t count, size_t stride, size_t component_stride, int components, int memspace)`
  - `int soma_set_communicator(soma_ctx, int fortran_comm)`
  - `int soma_initialize(soma_ctx)`
  - `int soma_begin_step(soma_ctx, const char* component, double* t_begin, double* t_end)`
  - `int soma_end_step(soma_ctx, const char* component)`
  - `int soma_requires_checkpoint(soma_ctx, const char* component, int* required)` and `soma_requires_restore(...)` — the same contract preCICE uses, exposed for components that cannot be driven by SOMA's loop
  - `int soma_run_period(soma_ctx)`, `int soma_run(soma_ctx, long periods)`
  - `int soma_is_coupling_ongoing(soma_ctx, int* ongoing)`
  - `int soma_last_error(soma_ctx, char* buffer, size_t buffer_size)` — the ABI carries no exceptions, so errors are retrieved rather than thrown
  - `const char* soma_version(void)`, `int soma_abi_version(void)`
  - Error codes as `#define SOMA_OK 0`, `SOMA_ERR_INVALID -1`, `SOMA_ERR_NOTFOUND -2`, `SOMA_ERR_STATE -3`, `SOMA_ERR_RUNTIME -4`, `SOMA_ERR_UNSUPPORTED -5`.
  - No exception may escape: every entry point wraps its body in `try`/`catch(...)` and converts to a code plus a stored message.

- [ ] **Step 1: Write the failing test — in C, not C++**

Compiling the test as C is what actually proves the header is C-compatible.

`test/unit/capi_test.c`:

```c
#include <soma/soma_c.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));   \
      ++failures;                                              \
    }                                                          \
  } while (0)

static void test_version(void) {
  CHECK(soma_version() != NULL, "version string is null");
  CHECK(strlen(soma_version()) > 0, "version string is empty");
  CHECK(soma_abi_version() >= 1, "abi version must be at least 1");
}

static void test_create_destroy(void) {
  soma_ctx ctx = NULL;
  CHECK(soma_create(&ctx) == SOMA_OK, "create failed");
  CHECK(ctx != NULL, "create returned a null context");
  CHECK(soma_destroy(ctx) == SOMA_OK, "destroy failed");
}

static void test_null_arguments_are_rejected(void) {
  CHECK(soma_create(NULL) == SOMA_ERR_INVALID, "null out-param accepted");
  CHECK(soma_destroy(NULL) == SOMA_ERR_INVALID, "null context accepted");
}

static void test_yaml_load_and_handles(void) {
  soma_ctx ctx = NULL;
  CHECK(soma_create(&ctx) == SOMA_OK, "create failed");

  const int rc = soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair.yaml");
  CHECK(rc == SOMA_OK, "yaml load failed");

  int handle = -1;
  CHECK(soma_field_handle(ctx, "fluid", "heat_flux", &handle) == SOMA_OK,
        "field handle lookup failed");
  CHECK(handle >= 0, "handle is negative");

  int missing = -1;
  CHECK(soma_field_handle(ctx, "fluid", "nope", &missing) == SOMA_ERR_NOTFOUND,
        "missing field should report NOTFOUND");

  soma_destroy(ctx);
}

static void test_register_field(void) {
  soma_ctx ctx = NULL;
  soma_create(&ctx);
  soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair.yaml");

  int handle = -1;
  soma_field_handle(ctx, "fluid", "heat_flux", &handle);

  double data[16];
  for (int i = 0; i < 16; ++i) data[i] = (double)i;
  const int rc = soma_register_field(ctx, handle, data, 16, 1, 0, 1, 0);
  CHECK(rc == SOMA_OK, "register_field failed");

  /* Re-registering must be refused, not silently accepted. */
  CHECK(soma_register_field(ctx, handle, data, 16, 1, 0, 1, 0) != SOMA_OK,
        "rebinding should be refused");

  soma_destroy(ctx);
}

static void test_error_message_is_retrievable(void) {
  soma_ctx ctx = NULL;
  soma_create(&ctx);
  soma_load_graph_yaml(ctx, "/definitely/not/a/file.yaml");

  char buffer[512];
  const int rc = soma_last_error(ctx, buffer, sizeof(buffer));
  CHECK(rc == SOMA_OK, "last_error failed");
  CHECK(strlen(buffer) > 0, "error message is empty");

  soma_destroy(ctx);
}

static void test_state_errors(void) {
  soma_ctx ctx = NULL;
  soma_create(&ctx);
  /* run before initialize */
  CHECK(soma_run_period(ctx) == SOMA_ERR_STATE, "run before init accepted");
  soma_destroy(ctx);
}

int main(void) {
  test_version();
  test_create_destroy();
  test_null_arguments_are_rejected();
  test_yaml_load_and_handles();
  test_register_field();
  test_error_message_is_retrievable();
  test_state_errors();

  if (failures == 0) {
    printf("capi_test: all checks passed\n");
    return 0;
  }
  printf("capi_test: %d failure(s)\n", failures);
  return 1;
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/soma_c.h` file not found.

- [ ] **Step 3: Write `include/soma/soma_c.h`**

Pure C99: no `namespace`, no default arguments, no `bool`, guarded with
`#ifdef __cplusplus extern "C" {`. Every function documented with its
preconditions and the error codes it can return.

- [ ] **Step 4: Write `src/capi/soma_c.cpp`**

`struct soma_ctx_s` holds the `CouplingGraph`, `OperatorDAG`, `ExecutionPlan`,
`FieldRegistry`, backend, `Runner`, a state enum, and `std::string last_error`.
Every entry point follows one shape:

```cpp
int soma_something(soma_ctx ctx, ...) {
  if (ctx == nullptr) return SOMA_ERR_INVALID;
  try {
    // ... real work, converting soma::Result to a code
  } catch (const std::exception& e) {
    ctx->last_error = e.what();
    return SOMA_ERR_RUNTIME;
  } catch (...) {
    ctx->last_error = "unknown error";
    return SOMA_ERR_RUNTIME;
  }
  return SOMA_OK;
}
```

- [ ] **Step 5: Register the C test**

```cmake
add_executable(capi_test unit/capi_test.c)
set_target_properties(capi_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_link_libraries(capi_test PRIVATE soma_core)
target_compile_definitions(capi_test PRIVATE
  SOMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME capi_test COMMAND capi_test)
```

The top-level `project()` needs `LANGUAGES CXX C`.

- [ ] **Step 6: Run**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R capi_test
```

Expected: `capi_test: all checks passed`.

- [ ] **Step 7: Commit**

```bash
git add include/soma/soma_c.h src/capi/ test/unit/capi_test.c \
        test/CMakeLists.txt src/CMakeLists.txt CMakeLists.txt
git commit -m "feat(soma): C ABI over the serialized graph narrow waist"
```

---

### Task 8.2: Fortran shim

Spec §6 mode 2. The shim ships and is unit-tested; a coupled `nek`/`diablo` run
is explicitly out of v1 scope (spec §1.2).

**Files:**
- Create: `fortran/soma_mod.f90`
- Create: `fortran/CMakeLists.txt`
- Create: `test/fortran/capi_test.f90`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: module `soma_mod` with `iso_c_binding` interfaces for every C entry point, plus Fortran-friendly wrappers that handle string termination and array descriptors:
  - `type :: soma_context` wrapping `type(c_ptr)`
  - `soma_f_create(ctx)`, `soma_f_destroy(ctx)`
  - `soma_f_load_yaml(ctx, path, ierr)` — appends `c_null_char`
  - `soma_f_field_handle(ctx, comp, field, handle, ierr)`
  - `soma_f_register_field(ctx, handle, array, ierr)` — takes a
    `real(c_double), target, contiguous :: array(:)` and derives count and
    stride, so callers never compute layout by hand
  - `soma_f_run(ctx, periods, ierr)`
  - `soma_f_last_error(ctx, message)`
  - Layout note in the module header: SOMA's `Layout` is element-major, and a
    contiguous Fortran 1-D array maps to `stride = 1`. A 2-D
    `array(ncomp, ncell)` maps to `stride = ncomp, component_stride = 1`; the
    wrapper provides an overload for that case.

- [ ] **Step 1: Write the failing test**

`test/fortran/capi_test.f90`:

```fortran
program soma_fortran_test
  use iso_c_binding
  use soma_mod
  implicit none

  type(soma_context) :: ctx
  integer            :: ierr, handle, failures
  real(c_double), target :: field(16)
  character(len=512) :: message
  integer            :: i

  failures = 0

  call soma_f_create(ctx, ierr)
  if (ierr /= 0) call fail('create failed', failures)

  call soma_f_load_yaml(ctx, SOMA_TEST_DATA_DIR//'/pair.yaml', ierr)
  if (ierr /= 0) then
     call soma_f_last_error(ctx, message)
     write(*,*) 'yaml load error: ', trim(message)
     call fail('yaml load failed', failures)
  end if

  call soma_f_field_handle(ctx, 'fluid', 'heat_flux', handle, ierr)
  if (ierr /= 0) call fail('field handle lookup failed', failures)
  if (handle < 0) call fail('handle is negative', failures)

  do i = 1, 16
     field(i) = real(i, c_double)
  end do

  call soma_f_register_field(ctx, handle, field, ierr)
  if (ierr /= 0) call fail('register_field failed', failures)

  ! A missing field must report an error rather than aborting.
  call soma_f_field_handle(ctx, 'fluid', 'nope', handle, ierr)
  if (ierr == 0) call fail('missing field should fail', failures)

  call soma_f_destroy(ctx, ierr)
  if (ierr /= 0) call fail('destroy failed', failures)

  if (failures == 0) then
     write(*,*) 'soma_fortran_test: all checks passed'
     stop 0
  else
     write(*,*) 'soma_fortran_test: ', failures, ' failure(s)'
     stop 1
  end if

contains

  subroutine fail(msg, count)
    character(len=*), intent(in)    :: msg
    integer,          intent(inout) :: count
    write(*,*) 'FAIL: ', msg
    count = count + 1
  end subroutine fail

end program soma_fortran_test
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — no module `soma_mod`.

- [ ] **Step 3: Write `fortran/soma_mod.f90` and `fortran/CMakeLists.txt`**

```cmake
enable_language(Fortran)

add_library(soma_fortran soma_mod.f90)
target_link_libraries(soma_fortran PUBLIC soma_core)
set_target_properties(soma_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
target_include_directories(soma_fortran PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/modules)
```

and in the top level, `if(SOMA_ENABLE_FORTRAN) add_subdirectory(fortran) endif()`.

- [ ] **Step 4: Register the test**

```cmake
if(SOMA_ENABLE_FORTRAN)
  add_executable(fortran_capi_test fortran/capi_test.f90)
  target_link_libraries(fortran_capi_test PRIVATE soma_fortran)
  target_compile_definitions(fortran_capi_test PRIVATE
    SOMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME fortran_capi_test COMMAND fortran_capi_test)
endif()
```

The `SOMA_TEST_DATA_DIR` macro in Fortran needs `-cpp`; add
`target_compile_options(fortran_capi_test PRIVATE -cpp)`.

- [ ] **Step 5: Run**

```bash
cmake -B build-fortran -DCMAKE_BUILD_TYPE=Debug -DSOMA_ENABLE_FORTRAN=ON
cmake --build build-fortran -j
ctest --test-dir build-fortran --output-on-failure -R fortran
```

Expected: `soma_fortran_test: all checks passed`.

- [ ] **Step 6: Commit**

```bash
git add fortran/ test/fortran/ CMakeLists.txt test/CMakeLists.txt
git commit -m "feat(soma): Fortran ISO_C_BINDING shim over the C ABI"
```

---

### Task 8.3: Python bindings

Spec §6: Python surrogates are peer components, and the Python front-end builds
the same graph object as C++ and YAML.

**Files:**
- Create: `python/CMakeLists.txt`
- Create: `python/src/bindings.cpp`
- Create: `python/soma/__init__.py`
- Create: `python/tests/test_graph.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces module `soma` exposing:
  - `Rational`, `Units`, `Region`, `FieldSpec`, `ComponentDecl`, `CouplingGraph`, `Channel` builder methods, enums `Discretization`, `Conservation`, `Intent`, `AccumOp`, `TemporalBasis`, `RemapFlags`
  - `soma.lower(graph)`, `soma.schedule(...)`, `soma.dump_ir(dag)`, `soma.dump_plan(plan)`
  - `graph.to_yaml()` / `soma.from_yaml(text)` / `soma.load_yaml(path)`
  - `soma.PyComponent` — a trampoline base class so a Python class can implement `register_fields`/`advance` and be a peer component; field data is exchanged as a NumPy array view over the component's storage, never a copy
  - Python interval literals accept `"1/3"`, `fractions.Fraction`, or `int`
  - Interpreter pinned to 3.11 to match the installed LowFive extension:
    `find_package(Python 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)`

- [ ] **Step 1: Write the failing test**

`python/tests/test_graph.py`:

```python
"""Python front-end must build the same graph object as C++ and YAML."""
from fractions import Fraction

import pytest

import soma


def build_pair() -> soma.CouplingGraph:
    g = soma.CouplingGraph()
    m1 = g.add_mesh("fluid_mesh", "generated:box:8x8x8")
    m2 = g.add_mesh("solid_mesh", "generated:box:5x5x5")
    r1 = g.add_region("fluid_wall", m1, entity_set="wall", dimension=2)
    r2 = g.add_region("solid_wall", m2, entity_set="wall", dimension=2)

    fluid = g.add_component("fluid", first_rank=0, rank_count=2,
                            interval="1/2")
    solid = g.add_component("solid", first_rank=2, rank_count=2, interval=1)

    q_out = g.add_field(fluid, "heat_flux", r1, units=soma.units.W_per_m2,
                        conservation=soma.Conservation.Extensive,
                        intent=soma.Intent.Provides)
    q_in = g.add_field(solid, "q_wall", r2, units=soma.units.W_per_m2,
                       conservation=soma.Conservation.Extensive,
                       intent=soma.Intent.Requires)

    ch = g.add_channel("flux", q_out, q_in)
    g.channel(ch).accumulate(soma.AccumOp.Mean).remap(
        soma.RemapFlags.Conservative | soma.RemapFlags.Monotone).relax(0.7)
    return g


def test_builds_a_graph():
    g = build_pair()
    assert g.component_count == 2
    assert g.channel_count == 1
    assert g.field_count == 2


def test_interval_accepts_string_fraction_and_int():
    g = soma.CouplingGraph()
    a = g.add_component("a", interval="1/3")
    b = g.add_component("b", interval=Fraction(1, 7))
    c = g.add_component("c", interval=2)
    assert g.component(a).interval == soma.Rational(1, 3)
    assert g.component(b).interval == soma.Rational(1, 7)
    assert g.component(c).interval == soma.Rational(2)


def test_yaml_round_trip():
    g = build_pair()
    text = g.to_yaml()
    restored = soma.from_yaml(text)
    assert soma.structurally_equal(g, restored)


def test_validation_reports_unit_mismatch():
    g = soma.CouplingGraph()
    m = g.add_mesh("m", "generated:box:2x2x2")
    r = g.add_region("r", m, dimension=3)
    a = g.add_component("a", first_rank=0, rank_count=1)
    b = g.add_component("b", first_rank=1, rank_count=1)
    s = g.add_field(a, "T", r, units=soma.units.K,
                    intent=soma.Intent.Provides)
    t = g.add_field(b, "q", r, units=soma.units.W_per_m2,
                    intent=soma.Intent.Requires)
    g.add_channel("bad", s, t)

    diags = soma.validate(g)
    assert diags.has_errors
    assert any(d.code == "E-UNITS" for d in diags.items)


def test_lowering_produces_an_ir_dump():
    g = build_pair()
    result = soma.lower(g)
    assert result.ok, result.diagnostics.to_string()
    text = soma.dump_ir(result.dag)
    assert text.startswith("# soma-ir v1")
    assert "Accumulate" in text
    assert "Remap" in text


def test_python_component_is_a_peer():
    """A Python class implementing the interface must couple like any other."""
    import numpy as np

    class Surrogate(soma.PyComponent):
        def __init__(self, n):
            super().__init__("surrogate")
            self.data = np.zeros(n)
            self.steps = 0

        def register_fields(self, registry, graph):
            cid = graph.find_component("surrogate")
            for fid in graph.component(cid).fields:
                registry.bind_numpy(fid, self.data)

        def advance(self, ctx):
            self.steps += 1
            self.data[:] = float(self.steps)

    c = Surrogate(8)
    assert c.name == "surrogate"
    assert c.steps == 0


def test_numpy_binding_is_a_view_not_a_copy():
    import numpy as np

    data = np.zeros(4)
    reg = soma.FieldRegistry()
    reg.bind_numpy(soma.FieldId(0), data)
    view = reg.get_numpy(soma.FieldId(0))
    view[1] = 7.5
    assert data[1] == 7.5, "the binding must alias the component's storage"


def test_rejects_non_contiguous_array():
    import numpy as np

    data = np.zeros((4, 3))[:, 0]     # strided view
    reg = soma.FieldRegistry()
    with pytest.raises(ValueError):
        reg.bind_numpy(soma.FieldId(0), data)
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `ModuleNotFoundError: No module named 'soma'`.

- [ ] **Step 3: Write the bindings and the CMake glue**

`python/CMakeLists.txt`:

```cmake
# Pinned to 3.11: the installed LowFive extension is built for 3.11 and Demo B
# loads both in one interpreter.
find_package(Python 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
  FIND_PACKAGE_ARGS NAMES pybind11)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_soma src/bindings.cpp)
target_link_libraries(_soma PRIVATE soma_core)

add_test(NAME python_tests
  COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests -v)
set_tests_properties(python_tests PROPERTIES
  ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_soma>:${CMAKE_CURRENT_SOURCE_DIR}")
```

`python/soma/__init__.py` re-exports from `_soma` and adds the pure-Python
conveniences (interval coercion from `Fraction`, a `units` namespace).

- [ ] **Step 4: Run**

```bash
cmake -B build-py -DCMAKE_BUILD_TYPE=Debug -DSOMA_ENABLE_PYTHON=ON \
      -DPython_EXECUTABLE=/opt/homebrew/bin/python3.11
cmake --build build-py -j
ctest --test-dir build-py --output-on-failure -R python_tests
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/ CMakeLists.txt
git commit -m "feat(soma): pybind11 bindings with NumPy field views"
```

---

### Task 8.4: LowFive transport

Spec §6 mode 4. Components are unmodified; coupling happens by intercepting
their HDF5 writes.

**Blocked if Task 8.0 Step 4 did not print `OK`.**

**Files:**
- Create: `src/transport/lowfive_transport.cpp`
- Modify: `include/soma/transport/transport.hpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/parallel/lowfive_transport_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `TransportSpec` gains `std::string outport; std::string inport;` — Wilkins's data-centric port model, where a port is an HDF5 path pattern such as `"out.h5/grid/temperature"` and wildcards like `"*.h5/particles"` are permitted.
  - `class LowFiveTransport final : public Transport` — registered as `"lowfive"`, `is_local()` false. `prepare` installs the VOL plugin and declares the intercepted datasets; `move` serves data from producer to consumer, which LowFive initiates when the producer closes the file and the consumer opens it.
  - `Result<void> set_communicator(int fortran_comm_handle)`.
  - Memory-mode by default (LowFive's in-memory MPI path); file mode available via `TransportSpec` for debugging, with a comment recording that file mode is orders of magnitude slower.

- [ ] **Step 1: Write the failing test**

`test/parallel/lowfive_transport_test.cpp` — two ranks, one writing an HDF5
dataset through plain HDF5 calls and one reading it, with no direct
communication between them. Tests: `InterceptsAWriteAndServesTheRead`,
`WildcardPortMatchesTheDataset`, `RegisteredInTheTransportRegistry`,
`ReportsAnErrorWhenTheVolCannotLoad`.

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `LowFiveTransport` is not declared.

- [ ] **Step 3: Implement**

Following the LowFive VOL API: construct `l5::DistMetadataVOL` with the
intercept-file and memory-file patterns, install it via `H5VLregister_connector`
and an `l5::H5VOLProperty`, and let the producer's `H5Fclose` trigger the serve.
DIY at `/opt/diy` supplies the block-parallel redistribution LowFive depends on.

- [ ] **Step 4: Run**

```bash
cmake -B build-lf -DCMAKE_BUILD_TYPE=Debug \
      -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_LOWFIVE=ON \
      -DCMAKE_PREFIX_PATH="/opt/lowfive-shared;/opt/hdf5-shared;/opt/diy"
cmake --build build-lf -j
ctest --test-dir build-lf --output-on-failure -R lowfive
```

- [ ] **Step 5: Commit**

```bash
git add src/transport/lowfive_transport.cpp include/soma/transport/transport.hpp \
        test/parallel/lowfive_transport_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): LowFive transport via HDF5 VOL interception"
```

---

### Task 8.5: Demo B — two unmodified executables

Spec §11 Demo B. The claim under test: intrusive and non-intrusive coupling
share one runtime.

**Files:**
- Create: `examples/lowfive_demo/producer.cpp`
- Create: `examples/lowfive_demo/consumer.cpp`
- Create: `examples/lowfive_demo/workflow.yaml`
- Create: `examples/lowfive_demo/run_demo.sh`
- Modify: `examples/CMakeLists.txt`
- Create: `test/verification/demo_b_test.cpp`

**Interfaces:**
- `producer.cpp` and `consumer.cpp` are **plain HDF5 programs**. They include
  `hdf5.h` and nothing from SOMA — that is the whole point, and a test asserts
  it by grepping their sources.
- `workflow.yaml` declares the coupling in Wilkins style, with `outport` and
  `inport` patterns, alongside the SOMA channel semantics.
- `run_demo.sh` launches both under `mpiexec` with the SOMA VOL active.
- `demo_b_test.cpp` asserts: the demo runs to completion; the consumer receives
  the producer's data bit-exactly; **neither source file mentions SOMA**; and
  the same `workflow.yaml` graph lowers to a DAG differing from Demo A's only in
  the `Transport` node's backend attribute — the concrete evidence for spec
  §2.2's claim.

- [ ] **Step 1: Write the failing acceptance test**

```cpp
// test/verification/demo_b_test.cpp  (sketch of the load-bearing assertions)

// The claim of spec §2.2, made checkable.
TEST(DemoB, SourcesDoNotMentionSoma) {
  for (const char* path : {SOMA_EXAMPLES_DIR "/lowfive_demo/producer.cpp",
                           SOMA_EXAMPLES_DIR "/lowfive_demo/consumer.cpp"}) {
    std::ifstream in{path};
    ASSERT_TRUE(in) << path;
    std::string text{std::istreambuf_iterator<char>{in}, {}};
    EXPECT_EQ(text.find("soma"), std::string::npos)
        << path << " must be an unmodified HDF5 program";
    EXPECT_NE(text.find("hdf5.h"), std::string::npos);
  }
}

TEST(DemoB, GraphDiffersFromDemoAOnlyInTransport) {
  const auto b = soma::load_yaml_file(
      SOMA_EXAMPLES_DIR "/lowfive_demo/workflow.yaml");
  ASSERT_TRUE(b.ok()) << b.error().message;
  const auto lowered = lower(b.value());
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  // Same operator kinds as an in-process channel of the same shape.
  const std::string ir = dump_text(lowered.dag);
  EXPECT_NE(ir.find("Transport"), std::string::npos);
  EXPECT_NE(ir.find("Remap"), std::string::npos);
}
```

- [ ] **Step 2: Write the demo programs, YAML, and script**

- [ ] **Step 3: Run**

```bash
cmake --build build-lf -j
./examples/lowfive_demo/run_demo.sh
ctest --test-dir build-lf --output-on-failure -R DemoB
```

- [ ] **Step 4: Commit**

```bash
git add examples/lowfive_demo/ examples/CMakeLists.txt \
        test/verification/demo_b_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): Demo B -- coupling two unmodified HDF5 executables"
```

---

## Phase 8 exit criteria

- [ ] The C ABI compiles under a **C99** compiler and no exception escapes it.
- [ ] The Fortran shim builds and its test passes; a coupled nek/diablo run remains out of v1 scope by design.
- [ ] The Python front-end builds a graph structurally equal to the C++ and YAML ones; NumPy bindings alias rather than copy.
- [ ] LowFive loads (`dlopen` returns non-null) and intercepts an HDF5 write.
- [ ] **Demo B** runs, and neither demo source mentions SOMA.
- [ ] All four `ctest` configurations green: `build`, `build-full`, `build-py`, `build-lf`.

---

## v1 completion checklist

- [ ] **Demo A** — DNWR pair: non-matching meshes, independent intervals, accelerator ladder, measured order ≈2, conservation to 1e-10.
- [ ] **Demo B** — two unmodified HDF5 executables coupled by interception.
- [ ] Core builds and tests with **zero** optional dependencies.
- [ ] No third-party type in any public header (`grep -rn "moab::\|YAML::\|MPI_\|hid_t" include/` finds nothing).
- [ ] Golden IR files current and byte-compared in CI.
- [ ] `docs/BUILDING.md` records every environment rebuild, with the "before" evidence.

**Deferred past v1, by design and stated in the spec:** JFNK/monolithic (§5.5),
Tier 2 4D intersection for moving meshes (§7.2), MPMD port-connect transport
(§6 mode 3), a running coupled Fortran physics demo (§1.2), GPU execution
beyond a device-pointer-clean design (§1.2).
