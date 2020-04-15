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

from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Flutes'
copyright = '2020, Zecong Hu'
author = 'Zecong Hu'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'recommonmark',
    'sphinxcontrib.spelling',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    # 'numpy': ('http://docs.scipy.org/docs/numpy/', None),
}

##### Customize ######

# Snippet to insert at beginning of each RST file.
rst_prolog = r"""
.. role:: python(code)
    :language: python
"""

autodoc_member_order = 'bysource'
autodoc_typehints = 'none'

spelling_lang = 'en_US'
spelling_word_list_filename = 'spelling_wordlist.txt'

master_doc = 'index'


def setup(app):
    with open("_index") as f:
        content = f.read()
    with open("../README.md") as f:
        readme = f.read()
        readme = "\n".join(readme.strip().split("\n")[1:])
    with open("index.md", "w") as f:
        content = content.replace("<REPLACE_WITH_README>", readme)
        f.write(content)

    app.add_config_value('recommonmark_config', {
        # 'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_math': False,
        'enable_inline_math': False,
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)
