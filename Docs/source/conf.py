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
sys.path.append("../..")
import torch
import deprecated
import torch_geometric
import torchmetrics
import os.path as osp
import pyg_sphinx_theme
import prompt_graph


# -- Project information -----------------------------------------------------

project = 'ProG'
copyright = '2024, WANGKevin'
author = 'WANGKevin'


# The full version, including alpha/beta/rc tags
release = 'v0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = [
#]

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'pyg',
]

html_theme = 'pyg_sphinx_theme'
html_logo = ('https://github.com/sheldonresearch/ProG/blob/main/Logo.jpg?raw=true')
html_favicon = ('https://github.com/sheldonresearch/ProG/blob/main/Logo.jpg?raw=true')
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    # 'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}

nbsphinx_thumbnails = {
    'tutorial/create_gnn':
    '_static/thumbnails/create_gnn.png',
    'tutorial/heterogeneous':
    '_static/thumbnails/heterogeneous.png',
    'tutorial/create_dataset':
    '_static/thumbnails/create_dataset.png',
    'tutorial/load_csv':
    '_static/thumbnails/load_csv.png',
    'tutorial/neighbor_loader':
    '_static/thumbnails/neighbor_loader.png',
    'tutorial/point_cloud':
    '_static/thumbnails/point_cloud.png',
    'tutorial/explain':
    '_static/thumbnails/explain.png',
    'tutorial/shallow_node_embeddings':
    '_static/thumbnails/shallow_node_embeddings.png',
    'tutorial/distributed_pyg':
    '_static/thumbnails/distributed_pyg.png',
    'tutorial/multi_gpu_vanilla':
    '_static/thumbnails/multi_gpu_vanilla.png',
    'tutorial/multi_node_multi_gpu_vanilla':
    '_static/thumbnails/multi_gpu_vanilla.png',
}


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, 'templates'):
        rst_context = {'torch_geometric': torch_geometric}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect('source-read', rst_jinja_render)
    app.add_js_file('js/version_alert.js')

def split_fullname(fullname):
    parts = fullname.split('.')
    classname = parts[-1]
    return classname