#!/usr/bin/env python

""" Logging related fixtures to aid testing.

.. code-author: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import pytest
import logging

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

@pytest.fixture
def logging_mixin(caplog):
    """ Logging mixin to capture logging messages from modules.

    It logs at the debug level, which is probably most useful for when a test fails.
    """
    caplog.set_level(loggingLevel)

