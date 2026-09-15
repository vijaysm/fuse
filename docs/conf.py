"""Sphinx configuration for the FUSE documentation.

Two rules keep this site honest:

1. Nothing here restates a value that lives in the code. Build options, lint
   rules and dependency pins are pulled in with ``literalinclude`` from the real
   files, so a doc page cannot drift from what the build actually does.
2. Warnings fail the build (``fail_on_warning`` in ``.readthedocs.yaml``), so a
   dangling cross-reference is caught in CI rather than read by a user.
"""

from __future__ import annotations

project = "FUSE"
author = "Vijay Mahadevan"
copyright = "2006-2026, Vijay Mahadevan"  # noqa: A001
release = "0.1.0.dev0"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",     # ::: fenced directives, readable in a plain editor
    "deflist",         # term/definition lists for vocabulary and error codes
    "attrs_inline",
    "substitution",
    "fieldlist",
]
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "FUSE"
html_theme_options = {
    "repository_url": "https://github.com/vijaysm/fuse",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
}

# Roles used across the guides.
rst_prolog = """
.. role:: cpp(code)
   :language: cpp
"""

# The API reference arrives at S1, when there is a public header to extract from.
# breathe_projects = {"fuse": "_doxygen/xml"}
# breathe_default_project = "fuse"
