#!/usr/bin/env python

""" Reaction plane fit package.

Despite being a python 3 package, we still need to include the ``__init__.py`` so
that we can locate files in the package with ``pkg_resources``.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

__all__ = [
    "base",
    "example",
    "fit",
    "functions",
    "three_orientations",
]

# Provide easy access to the version
# __version__ is the version string, while version_info is a tuple with an entry per point in the verion
from reaction_plane_fit.version import __version__   # noqa
from reaction_plane_fit.version import version_info  # noqa
