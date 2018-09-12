from setuptools import setup, find_packages
import versioneer

author = 'Florian R. Hölzlwimmer, David S. Fischer'

setup(
    name='diffxpy',
    author=author,
    author_email='david.fischer@helmholtz-muenchen.de',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'patsy',
        'batchglm>=0.2.0',
        'xarray',
        'statsmodels',
    ],
    extras_require={
        'optional': [
            'anndata',
        ],
        # 'scanpy_deps': [
        #     "scanpy",
        #     "anndata"
        # ],
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
