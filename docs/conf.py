# [MODULE DOC]
# Sphinx configuration for the Tunix documentation site. Executed by sphinx-build; defines module-level
# settings only: project metadata (name Tunix, copyright, author), extensions (myst_nb, sphinx_gallery,
# sphinx_collections, autodoc, autosummary, intersphinx, napoleon), templates path, exclude patterns,
# source suffixes (rst, md, ipynb), HTML theme options, sphinx-gallery and collections config,
# suppressed warnings, autodoc defaults (members, imported-members, undoc-members), intersphinx
# mappings for optax/flax/jax, and a setup hook that installs a log filter.
# Inputs: none (read by Sphinx). Outputs: module globals consumed by Sphinx. Side effects: setup() mutates
# the "sphinx" logger's warning handler.
# KNOWN ISSUE: in exclude_patterns two string literals (the from_safetensor_load glob and the rl README
# path) are adjacent without a comma, so they concatenate into one bogus pattern and neither is excluded.
"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import logging

from sphinx.util import logging as sphinx_logging

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tunix"
copyright = "2025, Tunix Developers"  # pylint: disable=redefined-builtin
author = "Tunix Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_gallery.gen_gallery",
    "sphinx_collections",
    # api docs
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    (
        "_collections/examples/model_load/from_safetensor_load/*"
        "_collections/examples/rl/README.md"
    ),
    "_collections/examples/sft/**",
    "_collections/examples/deepscaler/**",
]

source_suffix = [".rst", ".md", ".ipynb"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/img/tunix.png"
html_css_files = [
    "custom.css",
]

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/google/tunix",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
}

# -- Options for sphinx-gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "_collections/examples",  # path to your example scripts
    "gallery_dirs": (
        "_collections/gallery"
    ),  # path to where to save gallery generated output
    "filename_pattern": "*.py",
    "ignore_pattern": r"rl/|sft/|deepscaler/",
}

# -- Options for myst -------------------------------------------------------
myst_heading_anchors = 3
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "off"
nb_execution_allow_errors = False
nb_render_image_options = {}
nb_execution_excludepatterns = [
    "*.ipynb",
]

# -- Options for sphinx-collections

collections = {
    "examples": {
        "driver": "copy_folder",
        "source": "../examples",
        "ignore": [
            "model_load",
            "rl",
            "sft",
            "deepscaler",
        ],
    }
}


suppress_warnings = ["misc.highlighting_failure"]


# -- Options for the API reference

default_role = "py:obj"

napoleon_include_init_with_doc = False

autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "undoc-members": True,
}


intersphinx_mapping = {
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}


class FilterSphinxWarnings(logging.Filter):
  # [CLASS DOC]
  # logging.Filter that drops Sphinx log records whose message (stripped) starts with
  # "duplicate object description" (autosummary noise). Holds the Sphinx app object.
  """Filter autosummary 'duplicate object description' warnings.

  These warnings are unnecessary as they do not cause missing documentation
  or rendering issues, so it is safe to filter them out.
  """

  def __init__(self, app):
    # [FUNCTION DOC]
    # HELPER: stores the given Sphinx app on self.app, then calls the base Filter constructor (no name).
    # Inputs: app (Sphinx application). Outputs: none. Raises: nothing.
    self.app = app
    super().__init__()

  def filter(self, record: logging.LogRecord) -> bool:
    # [FUNCTION DOC]
    # Returns False (drop record) if the record's formatted message, stripped of whitespace, starts with
    # "duplicate object description"; otherwise True (keep). Inputs: record (LogRecord). Output: bool.
    # Raises: whatever record.getMessage raises on bad format args. No side effects.
    msg = record.getMessage()
    filter_out = ("duplicate object description",)
    return not msg.strip().startswith(filter_out)


def setup(app):
  # [FUNCTION DOC]
  # Sphinx extension hook. Gets the "sphinx" logger, finds handlers that are Sphinx WarningStreamHandler
  # instances, takes the first, and inserts a FilterSphinxWarnings(app) at position 0 of its filters.
  # Inputs: app (Sphinx application). Output: none. Raises ValueError (unpacking) if no
  # WarningStreamHandler is attached. Side effect: mutates global logging handler filters.
  """Set up custom logging filters."""
  logger = logging.getLogger("sphinx")
  warning_handler, *_ = [
      h
      for h in logger.handlers
      if isinstance(h, sphinx_logging.WarningStreamHandler)
  ]
  warning_handler.filters.insert(0, FilterSphinxWarnings(app))
