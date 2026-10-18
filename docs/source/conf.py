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
for optional in ("sphinx_copybutton", "sphinxext.opengraph", "sphinx_design"):
    try:
        __import__(optional)
        extensions.append(optional)
    except ImportError:  # optional extensions (docs/requirements.txt installs them)
        pass

# the heavy runtime dependencies are not needed to render the API pages
autodoc_mock_imports = ["torch", "triton", "mbirtorch"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = []
exclude_patterns = ["_build", "build"]
html_static_path = ["_static"]

# The logo: drop docs/source/_static/citrus_logo.png (and optionally citrus_logo_dark.png for the dark theme) into
# place and it is picked up; until then the theme shows the project name.
_static = os.path.join(os.path.dirname(__file__), "_static")
_logo = os.path.join(_static, "citrus_logo.png")
_logo_dark = os.path.join(_static, "citrus_logo_dark.png")

try:
    import sphinx_book_theme  # noqa: F401

    html_theme = "sphinx_book_theme"
    html_theme_options = {
        "repository_url": "https://github.com/cabouman/citrus",
        "use_repository_button": True,
        "use_download_button": False,
        "home_page_in_toc": True,
        "show_toc_level": 2,
        "logo": {"text": "CITRUS"},
    }
    if os.path.exists(_logo):
        html_logo = "_static/citrus_logo.png"
        html_theme_options["logo"] = {"image_light": "_static/citrus_logo.png",
                                      "image_dark": "_static/citrus_logo_dark.png" if os.path.exists(_logo_dark)
                                      else "_static/citrus_logo.png"}
except ImportError:
    html_theme = "alabaster"
_favicon = os.path.join(_static, "citrus_favicon.png")
if os.path.exists(_favicon):
    html_favicon = "_static/citrus_favicon.png"
html_title = "CITRUS"
html_short_title = "CITRUS"
ogp_site_url = "https://purduecitrus.readthedocs.io/"
ogp_image = "https://purduecitrus.readthedocs.io/en/latest/_static/citrus_pipeline.png"


def _literal_docstrings(app, what, name, obj, options, lines):
    """The docstrings are plain text with ASCII math (|x|, a_b, ...): render them verbatim."""
    if lines:
        text = list(lines)
        lines[:] = ["::", ""] + ["    " + line for line in text] + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _literal_docstrings)
