"""Sphinx configuration of the CITRUS documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "citrus"
author = "Jordan Houri and Charles A. Bouman"
copyright = "2026, Jordan Houri and Charles A. Bouman"


def _version() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "citrus", "__init__.py")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
    return "unknown"


version = release = _version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
for optional in ("sphinx_copybutton", "sphinxext.opengraph"):
    try:
        __import__(optional)
        extensions.append(optional)
    except ImportError:  # optional extensions (docs/requirements.txt installs them)
        pass

# the heavy runtime dependencies are not needed to render the API pages
autodoc_mock_imports = ["torch", "triton", "mbirtorch", "xrayphysics"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = []
exclude_patterns = ["_build", "build"]

try:
    import sphinx_book_theme  # noqa: F401

    html_theme = "sphinx_book_theme"
    html_theme_options = {
        "repository_url": "https://github.com/cabouman/citrus",
        "use_repository_button": True,
    }
except ImportError:
    html_theme = "alabaster"
html_title = "citrus"
ogp_site_url = "https://citrus.readthedocs.io/"


def _literal_docstrings(app, what, name, obj, options, lines):
    """The docstrings are plain text with ASCII math (|x|, a_b, ...): render them verbatim."""
    if lines:
        text = list(lines)
        lines[:] = ["::", ""] + ["    " + line for line in text] + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _literal_docstrings)
