import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "protoform"
copyright = "2026, Axl Cruz Garcia"
author = "Axl Cruz Garcia"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_thebe",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Intersphinx (cross-reference external projects) -----------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- Nitpick (suppress unavoidable broken references) ---------------------
# CorpusT is a protocol TypeVar — Sphinx cannot resolve it.
nitpick_ignore = [
    ("py:class", "CorpusT"),
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# -- sphinx-thebe (live notebook button) ------------------------------------
thebe_config = {
    "repository_url": "https://github.com/BitUnwiseOperator/protoform",
    "repository_branch": "main",
    "path_to_book": "docs/source",
}

# -- myst-nb ----------------------------------------------------------------
nb_execution_mode = "cache"
# Notebooks fetch from HuggingFace on first run — allow enough time.
nb_execution_timeout = 600

# -- Plotly -----------------------------------------------------------------
# Plotly figures output a <div> that requires the Plotly JS library to render.
# Alabaster does not load it by default — include it from the CDN so that
# interactive figures in notebook outputs appear in the baked HTML.
html_js_files = [
    (
        "https://cdn.plot.ly/plotly-2.35.2.min.js",
        {
            "integrity": (
                "sha384-cCVCZkAjYNxaYKbM8lsArLznDF"
                "/SvMFr1jcZrvOpSTCa0W40ZAdLzHCEulnUa5i7"
            ),
            "crossorigin": "anonymous",
        },
    ),
]
