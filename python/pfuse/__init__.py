"""pfuse: pybind11 bindings over fuse's C++ model and runtime.

See docs/dev/python-surface.md for the design. The compiled extension is `_pfuse`; this
module re-exports its names and adds the pure-Python conveniences that do not need C++
(mirroring upstream soma's own __init__.py/compiled-module split).
"""

from _pfuse import *  # noqa: F401,F403
from _pfuse import __doc__ as _doc  # noqa: F401

__all__ = [name for name in dir() if not name.startswith("_")]
