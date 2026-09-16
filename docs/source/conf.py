"""Sphinx configuration. The C extension is stubbed so the API reference builds without a GPU."""
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.abspath("../../python"))

# soliton._C loads libsoliton.so at import time, which does not exist on a docs builder.
_stub = types.ModuleType("soliton._C")
_stub.lib = mock.MagicMock()
_stub.longs = lambda xs: list(xs)
sys.modules["soliton._C"] = _stub

project = "Soliton"
copyright = "2026, Soliton contributors"
author = "Soliton contributors"
release = "0.0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
intersphinx_mapping = {"python": ("https://docs.python.org/3", None), "numpy": ("https://numpy.org/doc/stable", None)}

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["soliton.css"]
html_title = "Soliton documentation"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "icon_links": [{"name": "GitHub", "url": "https://github.com/vishesh9131/soliton", "icon": "fa-brands fa-github"}],
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "secondary_sidebar_items": ["page-toc"],
}
