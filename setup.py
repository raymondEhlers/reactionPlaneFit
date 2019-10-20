#!/usr/bin/env python

# Setup Jet-H analysis
# Derived from the setup.py in aliBuild and Overwatch
# and based on: https://python-packaging.readthedocs.io/en/latest/index.html

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
from typing import Any, cast, Dict

def get_version() -> str:
    version_module: Dict[str, Any] = {}
    with open(os.path.join("reaction_plane_fit", "version.py")) as f:
        exec(f.read(), version_module)
    return cast(str, version_module["__version__"])

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "reaction_plane_fit",
    version = get_version(),

    description = "Implments the reaction plane fit for background subtraction in heavy ion collisions.",
    long_description = long_description,
    long_description_content_type = "text/markdown",

    author = "Raymond Ehlers",
    author_email = "raymond.ehlers@cern.ch",

    url = "https://github.com/raymondEhlers/reactionPlaneFit",
    license = "BSD 3-Clause",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # Using typing (some need atleast 3.5+), and data classes (3.6+ with backport),
        # so we set the minimum value at 3.6.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords = 'HEP Physics',

    packages = find_packages(exclude = (".git", "tests")),

    # Rename scripts to the desired executable names
    # See: https://stackoverflow.com/a/8506532
    entry_points = {
    },

    # This is usually the minimal set of the required packages.
    install_requires=[
        "dataclasses;python_version<'3.7'",
        "numpy",
        "uproot",  # Handles ROOT I/O without needing ROOT itself.
        "iminuit",
        "pachyderm>2.2",
    ],

    # Include additional files
    include_package_data = True,
    package_data = {
        "reaction_plane_fit": ["py.typed"],
    },
    # Apparently this is required to be compatiable with mypy.
    # See: https://mypy.readthedocs.io/en/latest/installed_packages.html
    zip_safe = False,

    extras_require = {
        "tests": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "codecov",
            # For comparing to reference images
            "pytest-mpl",
        ],
        "docs": [
            "sphinx",
            # Allow markdown files to be used
            "recommonmark",
            # Allow markdown tables to be used
            "sphinx_markdown_tables",
        ],
        "dev": [
            "pre-commit",
            "flake8",
            "flake8-colors",
            "mypy",
        ]
    }
)
