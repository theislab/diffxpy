from setuptools import setup, find_packages
import versioneer

author = 'David S. Fischer, Florian R. Hölzlwimmer'
author_email='david.fischer@helmholtz-muenchen.de, diffxpy@frhoelzlwimmer.de'
description="Fast and scalable differential expression analysis on single-cell RNA-seq data"

with open("README.rst", "r") as fh:
     long_description = fh.read()

setup(
    name='diffxpy',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16.4',
        'scipy>=1.2.1',
        'pandas',
        'patsy>=0.5.0',
        'batchglm>=0.7.3',
        'statsmodels',
    ],
    extras_require={
        'optional': [
            'anndata',
        ],
        'plotting_deps': [
            "seaborn",
            "matplotlib"
        ],
        'docs': [
            'sphinx',
            'sphinx-autodoc-typehints',
            'sphinx_rtd_theme',
            'jinja2',
            'docutils',
        ],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
