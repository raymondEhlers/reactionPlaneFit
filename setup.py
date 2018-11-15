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
        # Using typing (some need atleast 3.5+), and data classes (3.7+),
        # so we set the minimum value at 3.7.
        'Programming Language :: Python :: 3.7',
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
            "flake8-colors"
        ]
    }
)
