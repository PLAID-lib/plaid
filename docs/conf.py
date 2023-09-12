# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, '.')
sys.path.insert(0, '../')

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)
sys.path.insert(0, os.path.join(basedir, "src/plaid"))
sys.path.insert(0, os.path.join(basedir, "tests"))
sys.path.insert(0, os.path.join(basedir, "examples"))
print(sys.path)


# -- Project information -----------------------------------------------------
root_doc = 'index'  # default is already <index>
project = 'plaid-lib'
copyright = '2023-{}, Safran'.format(datetime.date.today().year)
author = 'Safran'

# # The full version, including alpha/beta/rc tags
# release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.ifconfig',
    'sphinx.ext.duration',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'myst_nb',
    # 'myst_parser', # imported by myst_nb
    # 'sphinxcontrib.apidoc', # autoapi is better
    'sphinx.ext.autosummary',
]
# -----------------------------------------------------------------------------#
# sphinx.ext.intersphinx options
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pytest': ('https://pytest.org/en/stable/', None),
    'ipykernel': ('https://ipykernel.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('http://matplotlib.org/', None),
    # 'torch': ('https://pytorch.org/docs/stable/', None),
    # 'dgl': ('https://docs.dgl.ai/', None),
    # 'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
}
# sphinx.ext.extlinks options
extlinks_detect_hardcoded_links = True
# sphinx.ext.graphviz options
graphviz_output_format = 'svg'
# myst_parser options
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'myst-nb',
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify", # not installed
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 7  # max 7
# ---------------------------------------------------------#
# all those options are set in sphinx-apidoc commands in .gitlab-ci.yml
# # sphinxcontrib.apidoc options
# apidoc_module_dir = '../{}'.format(project)
# apidoc_output_dir = 'api_reference'
# apidoc_excluded_paths = ['tests']
# apidoc_extra_args = ['--templatedir=_templates/apidoc']
# apidoc_separate_modules = True
# apidoc_separate_modules = False
# ---------------------------------------------------------#
# autosummary options
autosummary_generate = True
# -----------------------------------------------------------------------------#
# autoapi options :
# https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html#setting-up-automatic-api-documentation-generation
extensions.append('autoapi.extension')
autoapi_dirs = ['../src/plaid']
# autoapi_dirs = ['../src/plaid', '../tests', '../examples']
autoapi_type = 'python'
autoapi_options = ['show-inheritance', 'show-module-summary', 'undoc-members']
# autoapi_options = ['show-inheritance', 'show-inheritance-diagram', 'show-module-summary', 'members']
# autoapi_options = ['show-inheritance', 'show-inheritance-diagram', 'show-module-summary', 'members', 'inherited-members', 'undoc-members', 'private-members', 'special-members', 'imported-members']
# 'members': Display children of an object
# 'inherited-members': Display children of an object that have been inherited from a base class.
# 'undoc-members': Display objects that have no docstring
# 'private-members': Display private objects (eg. _foo in Python)
# 'special-members': Display special objects (eg. __foo__ in Python)
# 'show-inheritance': Display a list of base classes below the class signature.
# 'show-inheritance-diagram': Display an inheritance diagram in generated class documentation. It makes use of the sphinx.ext.inheritance_diagram extension, and requires Graphviz to be installed.
# 'show-module-summary': Whether to include autosummary directives in generated module documentation.
# 'imported-members': Display objects imported from the same top level package or module. The default module template does not include imported objects, even with this option enabled. The default package template does.
# autodoc_typehints = "signature"
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_python_class_content = 'both'  # default is 'class'
# 'class': Use only the class docstring.
# 'both': Use the concatenation of the class docstring and the __init__ docstring.
# 'init': Use only the __init__ docstring.
# autoapi_member_order = 'bysource'
#     # 'alphabetical': Order members by their name, case sensitively.
#     # 'bysource': Order members by the order that they were defined in the source code.
#     # 'groupwise': Order members by their type then alphabetically, ordering the types as follows:
#     #     - Submodules and subpackages
#     #     - Attributes
#     #     - Exceptions
#     #     - Classes
#     #     - Functions
#     #     - Methods


# -----------------------------------------------------------------------------#

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
html_logo = "source/images/plaid.jpg"

# cf https://pradyunsg.me/furo/customisation/edit-button/
html_theme_options = {
    "source_edit_link": "https://gitlab.com/drti/plaid",
    # "source_repository": "https://gitlab.com/drti/plaid",
    "source_branch": "main",
    "source_directory": "docs/",
    # "source_directory": "docs/source",
    # "source_directory": "source",
    'logo_only': True,
}

github_url = 'https://gitlab.com/drti/plaid'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# A list of paths that contain extra files not directly related to the documentation,
# such as robots.txt or .htaccess.
# Relative paths are taken as relative to the configuration directory.
# They are copied to the output directory.
# They will overwrite any existing file of the same name.
# As these files are not meant to be built, they are automatically
# excluded from source files.
html_extra_path = ['_extra']

# -----------------------------------------------------------------------------#


def skip_logger_attribute(app, what, name, obj, skip, options):
    if what == 'data' and "logger" in name:
        print(f"WILL SKIP: {what=}, {name=}")
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_logger_attribute)
