#!/usr/bin/env python

""" Tests for the base module.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import os
import pytest
import numpy as np
import uproot

import reaction_plane_fit.base as base

logger = logging.getLogger(__name__)

@pytest.fixture
def setupHistogramConversion():
    """ Setup expected values for histogram conversion tests.

    This set of expected values corresponds to:

    >>> hist = ROOT.TH1F("test", "test", 10, 0, 10)
    >>> hist.Fill(3, 2)
    >>> hist.Fill(8)
    >>> hist.Fill(8)
    >>> hist.Fill(8)

    Note:
        The error on bin 9 (one-indexed) is just sqrt(counts), while the error on bin 4
        is sqrt(4) because we filled it with weight 2 (sumw2 squares this values).
    """
    expected = base.Histogram(x = np.arange(1, 11) - 0.5,
                              y = np.array([0, 0, 0, 2, 0, 0, 0, 0, 3, 0]),
                              errors_squared = np.array([0, 0, 0, 4, 0, 0, 0, 0, 3, 0]))

    histName = "rootHist"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testFiles", "convertHist.root")
    if not os.path.exists(filename):
        # Need to create the initial histogram.
        # This shouldn't happen very often, as the file is stored in the repository.
        import ROOT
        rootHist = ROOT.TH1F(histName, histName, 10, 0, 10)
        rootHist.Fill(3, 2)
        for _ in range(3):
            rootHist.Fill(8)

        # Write out with normal ROOT so we can avoid further dependencies
        fOut = ROOT.TFile(filename, "RECREATE")
        rootHist.Write()
        fOut.Close()

    return filename, histName, expected

def checkHist(inputHist: base.Histogram, expected: base.Histogram) -> bool:
    """ Helper function to compare a given Histogram against expected values.

    Args:
        inputHist (base.Histogram): Converted histogram.
        expected (base.Histogram): Expected hiostgram.
    Returns:
        bool: True if the histograms are the same.
    """
    h = base.Histogram.from_existing_hist(inputHist)
    np.testing.assert_allclose(h.x, expected.x)
    np.testing.assert_allclose(h.y, expected.y)
    np.testing.assert_allclose(h.errors, expected.errors)

    return True

@pytest.mark.root
def testROOTHistToHistogram(setupHistogramConversion):
    """ Check conversion of a read in ROOT file via ROOT to a Histogram object. """
    filename, histName, expected = setupHistogramConversion

    # Setup and read histogram
    import ROOT
    fIn = ROOT.TFile(filename, "READ")
    inputHist = fIn.Get(histName)

    assert checkHist(inputHist, expected) is True

def testUprootHistToHistogram(setupHistogramConversion):
    """ Check conversion of a read in ROOT file via uproot to a Histogram object. """
    filename, histName, expected = setupHistogramConversion

    # Retrieve the stored histogram via uproot
    uprootFile = uproot.open(filename)
    inputHist = uprootFile[histName]

    assert checkHist(inputHist, expected) is True

