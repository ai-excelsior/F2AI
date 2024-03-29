# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

docs_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.realpath(os.path.join(docs_dir, ".."))
sys.path.append(project_dir)
print(f'Append sys path: {project_dir}')

project = "F2AI"
copyright = "2022, eavae"
author = "eavae"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc"]
templates_path = ["templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_sidebars = {"**": ["globaltoc.html"]}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["static"]
