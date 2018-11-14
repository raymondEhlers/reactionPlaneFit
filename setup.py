#!/usr/bin/env python

# Setup Jet-H analysis
# Derived from the setup.py in aliBuild and Overwatch
# and based on: https://python-packaging.readthedocs.io/en/latest/index.html

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os

def getVersion():
    versionModule = {}
    with open(os.path.join("reactionPlaneFit", "version.py")) as f:
        exec(f.read(), versionModule)
    return versionModule["__version__"]

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="reactionPlaneFit",
    version=getVersion(),

    description="Implments the reaction plane fit for background subtraction in heavy ion collisions.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Raymond Ehlers",
    author_email="raymond.ehlers@cern.ch",

    url="https://github.com/raymondEhlers/reactionPlaneFit",
    license="BSD 3-Clause",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='HEP Physics',

    packages=find_packages(exclude=(".git", "tests")),

    # Rename scripts to the desired executable names
    # See: https://stackoverflow.com/a/8506532
    entry_points = {
    },

    # This is usually the minimal set of the required packages.
    # Packages should be installed via pip -r requirements.txt !
    install_requires=[
        "future",
        "aenum",
        "ruamel.yaml",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "uproot",  # Handles ROOT io without needing ROOT itself.
        #"rootpy",
        #"root_numpy",
        "numdifftools",
        "iminuit",
        "probfit",
    ],

    # Dependeny links are explained here: https://github.com/pypa/pip/issues/3610#issuecomment-283578756
    # I had to create forks to bump the version numbers. Otherwise, the metadata of the packages
    # (particularly the defined version number) mismatched the version selected here, which
    # caused constant warnings in pip (and gave confusing information where it said the requested
    # version was, for example, 1.2.1, but it installed 1.2. It actually installed 1.2.1, but the
    # metadata told it that it installed 1.2.). For users installing from pip, the best course of
    # action is for them to install the packages by hand before installing this package.
    dependency_links=[
        "git+https://github.com/raymondEhlers/probfit.git@master#egg=probfit-1.0.5.1"
    ],

    # Include additional files
    include_package_data=True,

    extras_require = {
        "tests": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "codecov",
        ],
        "docs": [
            "sphinx",
            # Allow markdown files to be used
            "recommonmark",
            # Allow markdown tables to be used
            "sphinx_markdown_tables",
        ],
        "dev": [
            "flake8",
        ]
    }
)
