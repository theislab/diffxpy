# This code was adapted from https://github.com/theislab/scanpy/scanpy/conf.py
# This file is therefore licensed under the license of the scanpy project,
# available from https://github.com/theislab/scanpy and copied here at the time of accession.
# Note that multiple changes were made to this file to adapt it to the diffxpy project.

# BSD 3-Clause License
#
# Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('agg')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import diffxpy


# -- General configuration ------------------------------------------------


needs_sphinx = '1.7'

# General information
project = 'diffxpy'
author = diffxpy.__author__
copyright = f'{datetime.now():%Y}, {author}.'
version = diffxpy.__version__.replace('.dirty', '')
release = version

# default settings
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
default_role = 'literal'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False

intersphinx_mapping = dict(
    anndata=('https://anndata.readthedocs.io/en/latest/', None),
    scanpy=('https://scanpy.readthedocs.io/en/latest/', None),
    numpy=('https://docs.scipy.org/doc/numpy/', None),
    pandas=('http://pandas.pydata.org/pandas-docs/stable/', None),
    python=('https://docs.python.org/3', None),
    scipy=('https://docs.scipy.org/doc/scipy/reference/', None)
)


# -- Options for HTML output ----------------------------------------------


html_theme = 'sphinx_rtd_theme'
html_theme_options = dict(
    navigation_depth=4,
    logo_only=True,           # Only show the logo
)
html_context = dict(
    display_github=True,      # Integrate GitHub
    github_user='theislab',   # Username
    github_repo='diffxpy',    # Repo name
    github_version='master',  # Version
    conf_py_path='/docs/',    # Path in the checkout to the docs root
)
html_static_path = ['_static']
html_show_sphinx = False
gh_url = 'https://github.com/{github_user}/{github_repo}'.format_map(html_context)


def setup(app):
    app.add_stylesheet('css/custom.css')
    app.connect('autodoc-process-docstring', insert_function_images)
    app.add_role('pr', autolink(f'{gh_url}/pull/{{}}', 'PR {}'))


# -- Options for other output formats ------------------------------------------


htmlhelp_basename = f'{project}doc'
doc_title = f'{project} Documentation'
latex_documents = [
    (master_doc, f'{project}.tex', doc_title, author, 'manual'),
]
man_pages = [
    (master_doc, project, doc_title, [author], 1)
]
texinfo_documents = [
    (master_doc, project, doc_title, author, project, 'One line description of project.', 'Miscellaneous'),
]


# -- Images for plot functions -------------------------------------------------


def insert_function_images(app, what, name, obj, options, lines):
    path = Path(__file__).parent / 'api' / f'{name}.png'
    if what != 'function' or not path.is_file(): return
    lines[0:0] = [f'.. image:: {path.name}', '   :width: 200', '   :align: right', '']


# -- GitHub links --------------------------------------------------------------


def autolink(url_template, title_template='{}'):
    from docutils import nodes

    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = url_template.format(text)
        title = title_template.format(text)
        node = nodes.reference(rawtext, title, refuri=url, **options)
        return [node], []
    return role
