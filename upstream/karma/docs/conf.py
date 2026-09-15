project = "KARMA"
copyright = "2007-2026, KARMA contributors"
release = "0.0.5"

extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "plans", "superpowers", "HANDOFF.md", "libmesh-to-moab-migration-plan.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
suppress_warnings = ["myst.header"]
