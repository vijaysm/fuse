#pragma once

// Shared by every binding file. See module.cpp's own header comment for the design these
// three declarations implement.

#include <pybind11/pybind11.h>

#include <string>

#include "fuse/base/result.hpp"

namespace pfuse_py {

/// Raises a bare pfuse.FuseError for a fuse::Result<T> Failure. Defined in module.cpp (the
/// only file that constructs pfuse exception objects); every other binding file calls this
/// rather than raising directly, so there is one raising path for the Result side of the
/// hierarchy.
[[noreturn]] void raise_result_failure(const std::string& code, const std::string& message);

/// `raise_result_failure` applied to a `fuse::Result<T>` that is known not to be ok(); callers
/// write `if (!r.ok()) pfuse_py::raise_for(r);` rather than repeating `.error().code`/
/// `.message()`
/// at every call site.
template <class T>
[[noreturn]] void raise_for(const fuse::Result<T>& r) {
  raise_result_failure(r.error().code, r.error().message);
}

/// Session.initialize()'s own validate/lower/schedule check: raises a bare pfuse.FuseError
/// carrying a bound fuse::Diagnostics (a materially different payload than a single
/// Failure{code, message} -- see module.cpp's header comment).
[[noreturn]] void raise_validation_failure(const std::string& what, pybind11::object diagnostics);

/// The pfuse.InvalidArgument / pfuse.Refusal type objects, for binding code that wants to
/// raise one of the six fuse::Error-mirroring classes directly (python/src/field_bindings.cpp's
/// own numpy buffer-contract checks use fuse::InvalidArgument via FUSE_THROW instead, which
/// reaches the SAME two classes through the registered translator -- these accessors exist
/// for the few places raising via FUSE_THROW is not convenient, e.g. inside a pybind11 argument
/// converter that fuse's own headers cannot see).
pybind11::handle invalid_argument_exc();
pybind11::handle refusal_exc();

}  // namespace pfuse_py
