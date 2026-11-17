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

root_doc = "contents"

exclude_patterns = ["index.md", "dev/reviews/**", "_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
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

# -- PDF (LaTeX) output -------------------------------------------------------
#
# Read the Docs builds the PDF with pdflatex, whose UTF-8 input encoding knows
# only the code points that a loaded package has declared. Sphinx translates a
# number of symbols itself (the right arrow, the check mark, superscript
# digits); every other non-ASCII symbol stops pdflatex with "Unicode character
# ... not set up for use with LaTeX" and fails the PDF build, while the HTML
# build passes.
#
# The table below declares each code point that the existing pages contain and
# that neither pdflatex nor Sphinx defines. A row is
#
#     (code point, LaTeX text-mode replacement, ASCII text for PDF bookmarks)
#
# New Markdown is written in ASCII, so this table is not expected to grow. A
# page that introduces a code point missing here fails the PDF build with the
# error quoted above; the remedy is to write the symbol in ASCII, or to add a
# row.
_m = r"\ensuremath{%s}"
_latex_unicode = [
    # Greek letters (set in math italic; no Greek text font is needed).
    ("0394", _m % r"\Delta", "Delta"),
    ("03A3", _m % r"\Sigma", "Sigma"),
    ("03B4", _m % r"\delta", "delta"),
    ("03B5", _m % r"\varepsilon", "epsilon"),
    ("03B7", _m % r"\eta", "eta"),
    ("03BB", _m % r"\lambda", "lambda"),
    ("03C9", _m % r"\omega", "omega"),
    # Superscript and subscript letters and signs.
    ("1D40", r"\textsuperscript{T}", "^T"),
    ("1D47", r"\textsuperscript{b}", "^b"),
    ("207B", r"\textsuperscript{\ensuremath{-}}", "^-"),
    ("2090", r"\textsubscript{a}", "_a"),
    ("2C7C", r"\textsubscript{j}", "_j"),
    # Arrows.
    ("2194", _m % r"\leftrightarrow", "<->"),
    ("21C4", _m % r"\rightleftarrows", "<->"),
    ("21D2", _m % r"\Rightarrow", "=>"),
    # Operators and relations.
    ("2212", _m % "-", "-"),
    ("2218", _m % r"\circ", "o"),
    ("221A", _m % r"\surd", "sqrt"),
    ("2229", _m % r"\cap", "cap"),
    ("222B", _m % r"\textstyle\int", "int"),
    ("222C", _m % r"\textstyle\iint", "iint"),
    ("2248", _m % r"\approx", "~="),
    ("2260", _m % r"\neq", "!="),
    ("2264", _m % r"\leq", "<="),
    ("2265", _m % r"\geq", ">="),
    ("2297", _m % r"\otimes", "(x)"),
    ("2308", _m % r"\lceil", "ceil("),
    ("2309", _m % r"\rceil", ")"),
    # Geometric shape used as an arrow head in a text diagram.
    ("25B6", _m % r"\blacktriangleright", ">"),
]

latex_elements = {
    "preamble": "\n".join(
        [
            r"\makeatletter",
            # Inside monospaced text (literal blocks, inline code) a replacement
            # is centred in a box one character wide, so that text diagrams and
            # aligned columns keep their alignment.
            r"\edef\fuse@ttfamily{\ttdefault}",
            r"\DeclareRobustCommand{\fuseunichar}[1]{%",
            r"  \ifx\f@family\fuse@ttfamily",
            r"    \leavevmode\hbox to\fontcharwd\font`0{\hss#1\hss}%",
            r"  \else#1\fi}",
            # U+0303 COMBINING TILDE follows its base letter in the input, so
            # the accent is set over the width of the preceding letter 'x'.
            r"\DeclareRobustCommand{\fusecombiningtilde}{%",
            r"  \begingroup\setbox\z@\hbox{x}%",
            r"  \llap{\hbox to\wd\z@{\hss\~{}\hss}}\endgroup}",
            r"\makeatother",
            r"\DeclareUnicodeCharacter{0303}"
            r"{\texorpdfstring{\fusecombiningtilde}{\textasciitilde}}",
            # Box-drawing corners: Sphinx declares only U+2500, U+2502, U+2514
            # and U+251C. pmboxdraw declares the whole block for inputenc.
            r"\IfFileExists{pmboxdraw.sty}{\usepackage{pmboxdraw}}{%",
            r"  \DeclareUnicodeCharacter{250C}{+}%",
            r"  \DeclareUnicodeCharacter{2510}{+}%",
            r"  \DeclareUnicodeCharacter{2518}{+}}",
        ]
        + [
            r"\DeclareUnicodeCharacter{%s}{\texorpdfstring{\fuseunichar{%s}}{%s}}"
            % row
            for row in _latex_unicode
        ]
    ),
}

# -- Root index page ----------------------------------------------------------
#
# Read the Docs fails a build whose HTML output has no index.html at its top
# level. The root document here is contents.md (docs/index.md is the landing
# page for the repository and the wiki, and is excluded above), so the HTML
# builders write contents.html and no index.html. The hook below adds an
# index.html that redirects to the root document. It is derived from
# ``root_doc`` and is skipped if a page named index is ever built.
_ROOT_REDIRECT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
</head>
<body>
<p>The documentation starts at <a href="{target}">{target}</a>.</p>
</body>
</html>
"""


def _write_root_redirect(app, exception):
    from pathlib import Path

    if exception is not None or app.builder.name not in ("html", "dirhtml"):
        return
    index = Path(app.outdir) / "index.html"
    if index.exists():
        return
    target = app.builder.get_target_uri(app.config.root_doc)
    index.write_text(
        _ROOT_REDIRECT.format(title=app.config.html_title, target=target),
        encoding="utf-8",
    )


def setup(app):
    app.connect("build-finished", _write_root_redirect)


# Optional API reference from Doxygen XML (requires breathe).
# breathe_projects = {"fuse": "_doxygen/xml"}
# breathe_default_project = "fuse"
