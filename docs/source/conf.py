# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



project = 'Neurocat'
copyright = '2025, Saiwen Yu'
author = 'Saiwen Yu'
release = '0.0.1'

# Import project

import sys
import os

sys.path.insert(0, os.path.join(os.path.pardir, os.path.pardir, 'src'))
import neurocat  # noqa

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

language = 'en'

extensions = [
    'sphinx.ext.autodoc',  # Automatically generates documentation from docstrings
    'sphinx.ext.autosummary',  # Generates summary tables for modules and classes
    'sphinx_autodoc_typehints',  # beautify the function
    'sphinx.ext.napoleon',  # Supports Google and NumPy style docstrings
    'sphinx_copybutton',  # Adds a copy button to code blocks
    'sphinx.ext.mathjax',  # Enables MathJax for rendering mathematical expressions
    'sphinx.ext.viewcode',  # Adds links to source code in documentation
    #'nbsphinx',  # Generates galleries for examples
    
    
]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

autodoc_typehints = 'description'

# Generate the API documentation when building
autosummary_generate = True
autodoc_default_options = {'members': True, 'inherited-members': True}
numpydoc_show_class_members = True
autoclass_content = "class"

templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



# nbsphinx_execute = 'never'