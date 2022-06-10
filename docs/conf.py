# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
package_path = os.path.abspath("..")
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))


# -- Project information -----------------------------------------------------

project = "SaQC"
copyright = (
    "2020, Bert Palm, David Schäfer, Peter Lünenschloß, Lennart Schmidt, Juliane Geller"
)
author = "Bert Palm, David Schäfer, Peter Lünenschloß, Lennart Schmidt, Juliane Geller"

# The full version, including alpha/beta/rc tags
release = f"2.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    # "sphinx.ext.extlinks",
    # "sphinx.ext.todo",
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.coverage",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.ifconfig",
    "sphinx.ext.autosectionlabel",
    # link source code
    "sphinx.ext.viewcode",
    # add suupport for NumPy style docstrings
    "sphinx.ext.napoleon",
    # Doc a whole module
    # see https://sphinx-automodapi.readthedocs.io/en/latest/
    "sphinx_automodapi.automodapi",
    # 'sphinx_automodapi.smart_resolver',
    # see https://sphinxcontrib-fulltoc.readthedocs.io/en/latest/
    "sphinxcontrib.fulltoc",
    # Markdown sources support
    # https://recommonmark.readthedocs.io/en/latest/
    "recommonmark",
    # https://github.com/ryanfox/sphinx-markdown-tables
    "sphinx_markdown_tables",
    # snippet plotting
    "matplotlib.sphinxext.plot_directive",
    # jupyter code execution
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    # "numpydoc"
]


# -- Params of the extensions ------------------------------------------------
autosummary_ignore_module_all = True
autosummary_imported_members = False
add_module_names = False
numpydoc_show_class_members = False
plot_html_show_formats = False
plot_html_show_source_link = False
automodsumm_inherited_members = True
# write out the files generated by automodapi, mainly for debugging
automodsumm_writereprocessed = True
automodapi_writereprocessed = True
automodapi_inheritance_diagram = False
automodapi_toctreedirnm = "_api"
autosectionlabel_prefix_document = True

autodoc_typehints = "none"

doctest_global_setup = """
import saqc
import pandas as pd
import numpy as np
from saqc.constants import *
"""
# -- Other options -----------------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "nature"

# use pandas theme
# html_theme = "pydata_sphinx_theme"


# html_theme_options = {
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- RST options -------
rst_prolog = """
.. |ufzLogo| image:: /resources/images/Representative/UFZLogo.png
   :width: 40 %
   :target: https://www.ufz.de/

.. |rdmLogo| image:: /resources/images/Representative/RDMLogo.png
   :width: 22 %
   :target: https://www.ufz.de/index.php?de=45348
   :class: align-right

|ufzLogo| |rdmLogo|
"""
