// pfuse's entry point: PYBIND11_MODULE(_pfuse, m), the exception hierarchy and translator, and
// the calls into each layer's own bind_*(m) function.
//
// -----------------------------------------------------------------------------------------
// ERRORS (docs/dev/python-surface.md, "Errors: two sources, one Python hierarchy")
// -----------------------------------------------------------------------------------------
// fuse reports failure two ways, both of which must become a Python exception:
//
//   1. fuse::Result<T> -- a Failure{code, message} returned, never thrown. There is no closed
//      enum behind the code string (unlike soma's 5-value SOMA_ERR_*), so every such failure becomes a BARE
//      pfuse.FuseError carrying .code (the E-* string) and .message (the text);
//      there is no hand triage into more subclasses. Binding code that calls something returning fuse::Result<T> and gets a
//      Failure raises `raise_result_failure(f)` (below).
//
//   2. fuse::Error (FUSE_ASSERT/FUSE_THROW/FUSE_REFUSE) -- an exception thrown across the
//      call. pybind11's py::register_exception_translator below catches `const fuse::Error&`
//      and re-raises the matching one of SIX Python subclasses, mirroring fuse::Error's own
//      C++ hierarchy exactly (error.hpp): InvariantViolation, InvalidArgument (NullPointer is
//      its own subclass, exactly as error.hpp derives NullPointer from InvalidArgument, not
//      from Error directly), Refusal, NotImplementedFeature (named to avoid shadowing the
//      builtin NotImplementedError; it DERIVES from it instead, which is not shadowing),
//      IoError. Binding-level precondition checks (python/src/field_bindings.cpp's own numpy
//      buffer-contract checks, "not found" lookups wrapping a C++ std::optional/pointer into
//      a raise) are written using FUSE_THROW(fuse::InvalidArgument, ...) directly -- the SAME
//      macro the library itself uses -- so they are translated by this ONE mechanism too,
//      rather than a second, parallel raising path.
//
// Every one of the six classes derives from pfuse.FuseError AND, except InvariantViolation
// (deliberately -- see its own comment below), from the closest Python builtin, so
// `except ValueError:` keeps working for a caller who never imports pfuse's own hierarchy.
//
// fuse::InvariantViolation is only ever SEEN as a Python exception under
// FUSE_ON_ERROR=throw|backtrace (the library default, which throws after an optional print).
// Under FUSE_ON_ERROR=abort, fuse::detail::raise() calls std::abort() before ever throwing;
// pfuse does not call fuse::set_error_policy() and does not change that.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include <Kokkos_Core.hpp>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"

namespace py = pybind11;

namespace pfuse_py {

void bind_base(py::module_& m);
void bind_model(py::module_& m);
void bind_field(py::module_& m);
void bind_component(py::module_& m);

// ---------------------------------------------------------------------------------------
// Exception objects, held for the interpreter's lifetime. A `py::object` at namespace scope
// is deliberately avoided (its destructor would run at static-destruction time, after the
// interpreter may already be finalized) -- the same reasoning upstream's own bindings.cpp
// documents for its ExcTypes; a leaked py::handle to a module-level exception class is the
// standard, correct trade.
struct ExcTypes {
  py::handle fuse_error;
  py::handle invariant_violation;
  py::handle invalid_argument;
  py::handle null_pointer;
  py::handle refusal;
  py::handle not_implemented_feature;
  py::handle io_error;
};

ExcTypes& exc() {
  static ExcTypes e;
  return e;
}

py::handle make_exception(py::module_& m, const char* name,
                          const std::vector<py::handle>& bases, const char* doc) {
  py::tuple base_tuple(bases.size());
  for (std::size_t i = 0; i < bases.size(); ++i) {
    base_tuple[i] = py::reinterpret_borrow<py::object>(bases[i]);
  }
  const std::string qualified = std::string("pfuse.") + name;
  PyObject* raw = PyErr_NewExceptionWithDoc(qualified.c_str(), doc, base_tuple.ptr(), nullptr);
  if (raw == nullptr) throw py::error_already_set();
  py::object obj = py::reinterpret_steal<py::object>(raw);
  m.add_object(name, obj);
  return py::handle(obj.release());  // intentionally leaked: see ExcTypes's own comment
}

/// Populates a freshly-constructed pfuse.FuseError (or subclass) instance's attributes from a
/// fuse::Error. Shared by every arm of the translator below so the five attribute names
/// (.code/.message/.where/.rank/.backtrace) are set identically regardless of which of the six
/// subclasses actually fired.
void fill_from_fuse_error(py::object& inst, const fuse::Error& e) {
  inst.attr("code") = py::cast(std::string(e.code()));
  inst.attr("message") = py::cast(std::string(e.message()));
  const auto& where = e.where();
  inst.attr("where") = py::cast(std::string(where.file_name()) + ":" +
                                std::to_string(where.line()));
  inst.attr("rank") = (e.rank() >= 0) ? py::cast(e.rank()) : py::none();
  inst.attr("backtrace") = e.backtrace().empty()
                               ? py::object(py::none())
                               : py::cast(e.backtrace().format());
  inst.attr("diagnostics") = py::none();
}

[[noreturn]] void raise_fuse_error(py::handle type, const fuse::Error& e) {
  py::object inst = py::reinterpret_borrow<py::object>(type)(std::string(e.what()));
  fill_from_fuse_error(inst, e);
  PyErr_SetObject(type.ptr(), inst.ptr());
  throw py::error_already_set();
}

}  // namespace pfuse_py

namespace pfuse_py {

/// Raises a bare pfuse.FuseError for a fuse::Result<T> Failure -- see this file's own header
/// comment for why there is no further subclassing here. Declared in errors.hpp, which every
/// binding file includes; defined here once so only module.cpp constructs the exception object.
[[noreturn]] void raise_result_failure(const std::string& code, const std::string& message) {
  py::object inst = py::reinterpret_borrow<py::object>(exc().fuse_error)(
      "[" + code + "] " + message);
  inst.attr("code") = py::cast(code);
  inst.attr("message") = py::cast(message);
  inst.attr("where") = py::none();
  inst.attr("rank") = py::none();
  inst.attr("backtrace") = py::none();
  inst.attr("diagnostics") = py::none();
  PyErr_SetObject(exc().fuse_error.ptr(), inst.ptr());
  throw py::error_already_set();
}

/// Raises a bare pfuse.FuseError carrying a bound fuse::Diagnostics (Session.initialize()'s
/// own validate/lower/schedule check -- a materially different payload shape than a single
/// Failure{code, message}, which is why it is not folded into raise_result_failure() above;
/// see docs/dev/python-surface.md's "Errors" section for the reasoning).
[[noreturn]] void raise_validation_failure(const std::string& what, py::object diagnostics) {
  py::object inst = py::reinterpret_borrow<py::object>(exc().fuse_error)(what);
  inst.attr("code") = py::none();
  inst.attr("message") = py::cast(what);
  inst.attr("where") = py::none();
  inst.attr("rank") = py::none();
  inst.attr("backtrace") = py::none();
  inst.attr("diagnostics") = std::move(diagnostics);
  PyErr_SetObject(exc().fuse_error.ptr(), inst.ptr());
  throw py::error_already_set();
}

py::handle invalid_argument_exc() { return exc().invalid_argument; }
py::handle refusal_exc() { return exc().refusal; }

}  // namespace pfuse_py

PYBIND11_MODULE(_pfuse, m) {
  m.doc() = "pybind11 bindings over fuse's C++ model and runtime (docs/dev/python-surface.md)";

  // fuse::Runner's coupling hot loop runs on Kokkos' default host execution space (every
  // existing C++ driver -- test/support/fuse_test_main.cpp, examples/*/main.cpp -- creates a
  // Kokkos::ScopeGuard before touching a Runner). A Python process has no equivalent of
  // main()'s own argc/argv lifetime to tie a ScopeGuard to, and Kokkos::ScopeGuard's
  // destructor aborts if Kokkos was already finalized -- so, deliberately, Kokkos is
  // initialized once here with its own defaults and NEVER finalized: the intentional leak
  // this file's own ExcTypes already accepts for the same reason (a static destructor that
  // might run after the interpreter has begun finalizing is a worse failure mode than
  // skipping a cleanup the OS reclaims at process exit anyway).
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::InitializationSettings settings;
    Kokkos::initialize(settings);
    // Explicitly finalize at process exit via std::atexit, RATHER than leaving Kokkos'
    // own global destructors to run uncontrolled after the Python interpreter has begun
    // tearing down its own threading state: measured directly that omitting this produced a "mutex lock failed" abort message during
    // interpreter shutdown even on an all-tests-pass run (harmless to the exit code, but
    // a real teardown-ordering defect this avoids rather than papers over).
    std::atexit([]() {
      if (Kokkos::is_initialized()) Kokkos::finalize();
    });
  }

  using namespace pfuse_py;

  // FuseError: the common base. Also used BARE for every fuse::Result<T> Failure (see this
  // file's header comment). RuntimeError is the closest builtin to fuse::Error's own base,
  // std::runtime_error.
  exc().fuse_error = make_exception(
      m, "FuseError", {PyExc_RuntimeError},
      "Base of every pfuse exception. .code (str, an E-* code, or None when the failure "
      "came from a thrown fuse::Error rather than a fuse::Result Failure), .message (str), "
      ".diagnostics (a bound fuse.Diagnostics, or None), .where/.rank/.backtrace (str/int/"
      "str, or None, populated only when a C++ fuse::Error crossed the boundary).");

  // The six classes mirroring fuse::Error's own hierarchy (error.hpp), in the SAME
  // inheritance shape: NullPointer derives from InvalidArgument, not from FuseError
  // directly, exactly as fuse::NullPointer derives from fuse::InvalidArgument in C++.
  exc().invariant_violation = make_exception(
      m, "InvariantViolation", {exc().fuse_error},
      "fuse::InvariantViolation: a FUSE_ASSERT fired -- the library's own logic is wrong, "
      "never a caller mistake. Deliberately NOT a subclass of any builtin exception (unlike "
      "the other five): catching it with `except Exception:` should take deliberate care, not "
      "an incidental `except ValueError:` written for a different reason. Only ever seen "
      "under FUSE_ON_ERROR=throw|backtrace; under FUSE_ON_ERROR=abort the process terminates "
      "before this exception exists.");
  exc().invalid_argument = make_exception(
      m, "InvalidArgument", {exc().fuse_error, PyExc_ValueError},
      "fuse::InvalidArgument: the caller broke a documented precondition.");
  exc().null_pointer = make_exception(
      m, "NullPointer", {exc().invalid_argument},
      "fuse::NullPointer: a null where an object was required -- a specific InvalidArgument, "
      "mirroring error.hpp's own `class NullPointer : public InvalidArgument`.");
  exc().refusal = make_exception(
      m, "Refusal", {exc().fuse_error, PyExc_ValueError},
      "fuse::Refusal: a deliberate rejection under invariant I4, carrying the E-code a test "
      "asserts on.");
  exc().not_implemented_feature = make_exception(
      m, "NotImplementedFeature", {exc().fuse_error, PyExc_NotImplementedError},
      "fuse::NotImplemented: a declared interface with no implementation behind it yet. "
      "Named NotImplementedFeature, not NotImplementedError, so importing pfuse's names does "
      "not shadow the builtin of the same name -- it DERIVES from that builtin instead, which "
      "is not shadowing.");
  exc().io_error = make_exception(
      m, "IoError", {exc().fuse_error, PyExc_OSError},
      "fuse::IoError: file or stream access failed; .message names the path.");

  py::register_exception_translator([](std::exception_ptr p) {
    if (!p) return;
    try {
      std::rethrow_exception(p);
    } catch (const fuse::InvariantViolation& e) {
      raise_fuse_error(exc().invariant_violation, e);
    } catch (const fuse::NullPointer& e) {
      raise_fuse_error(exc().null_pointer, e);
    } catch (const fuse::InvalidArgument& e) {
      raise_fuse_error(exc().invalid_argument, e);
    } catch (const fuse::Refusal& e) {
      raise_fuse_error(exc().refusal, e);
    } catch (const fuse::NotImplemented& e) {
      raise_fuse_error(exc().not_implemented_feature, e);
    } catch (const fuse::IoError& e) {
      raise_fuse_error(exc().io_error, e);
    } catch (const fuse::Error& e) {
      // Any fuse::Error subtype this list does not (yet) name individually -- defensive, not
      // expected to be reached: error.hpp declares exactly the six above.
      raise_fuse_error(exc().fuse_error, e);
    }
  });

  // The diagnostics layer's process-wide set-up (fuse/base/error.hpp): the
  // FUSE_ON_ERROR family is read here, at import, so the behaviour this
  // file's own comments describe (abort under FUSE_ON_ERROR=abort, throw
  // under =throw|backtrace) is what the interpreter actually gets. After
  // the translator above, so an unrecognised value surfaces as the
  // InvalidArgument it is rather than an untyped import failure. No rank: a
  // Python process drives fuse::SerialMeshBackend with no communicator
  // (docs/dev/python-surface.md, "What this does not expose").
  fuse::initialize();

  bind_base(m);
  bind_model(m);
  bind_field(m);
  bind_component(m);
}
