# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../src/subhkl'))

import inspect
import importlib

from typing import Any, Union
import sphinx.config


def add(self, name: str, default: Any, rebuild: Union[bool, str], types: Any) -> None:
    self.values[name] = (default, rebuild, types)


sphinx.config.Config.add = add

import subhkl
import subhkl._version

# -- Project information -----------------------------------------------------

project = 'subhkl'
copyright = '2024, Zachary Morgan'
author = 'Zachary Morgan'
version = subhkl._version.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

html_theme = 'pydata_sphinx_theme'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_title = 'subhkl'

html_sidebars = {'**': []}

# -- Extension configuration -------------------------------------------------

plot_pre_code = """
import numpy as np
np.random.seed(13)

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.ioff()

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)

"""

plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False
plot_basedir = ''

add_module_names = False


def linkcode_resolve(domain, info):
    baseurl = 'https://github.com/zjmorgan/subhkl/blob/main/{}.py'
    if 'py' not in domain:
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    url = baseurl.format(filename)
    mod = importlib.import_module(info['module'])
    if hasattr(mod, '__pyx_unpickle_Enum'):
        url += 'x'
    objname, *attrname = info['fullname'].split('.')
    obj = getattr(mod, objname)
    if attrname:
        for attr in attrname:
            obj = getattr(obj, attr)
    if not hasattr(mod, '__pyx_unpickle_Enum'):
        lines = inspect.getsourcelines(obj)
        start, stop = lines[1], lines[1] + len(lines[0]) - 1
        return '{}#L{}-L{}'.format(url, start, stop)
    else:
        return url
