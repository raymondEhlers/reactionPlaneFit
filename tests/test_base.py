#!/usr/bin/env python

# Tests for the various fit functions
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 17 June 2018

import pytest
import numpy as np
import uproot
import tempfile
import logging

import reactionPlaneFit.base as base

logger = logging.getLogger(__name__)

@pytest.fixture
def setupHistToArray():
    """ """
    # This simulates
    # hist.Fill(3, 2)
    # hist.Fill(8)
    # hist.Fill(8)
    # hist.Fill(8)
    # The error on bin 9 (one-indexed) is just sqrt(counts), while the error
    # on bin 4 is sqrt(4) because we filled it with weight 2 (sumw2 squares this values).
    expected = base.Histogram(x = np.arange(1, 11) - 0.5,
                              y = np.array([0, 0, 0, 2, 0, 0, 0, 0, 3, 0]),
                              errors = np.array([0, 0, 0, np.sqrt(4), 0, 0, 0, 0, np.sqrt(3), 0]))

    # ROOT hist
    # This is the only ROOT dependence in the package
    import rootpy.ROOT as ROOT
    import rootpy.io
    rootHist = ROOT.TH1F("rootHist", "rootHist", 10, 0, 10)
    rootHist.Fill(3, 2)
    for _ in range(3):
        rootHist.Fill(8)

    # Uproot rootio read hist
    with tempfile.NamedTemporaryFile() as f:
        clonedHistName = "rootHistInFile"
        with rootpy.io.root_open(f.name, "w"):
            rootHistForFile = rootHist.Clone(clonedHistName)
            rootHistForFile.Write()

        # Now retrieve the hist from that file
        uprootFile = uproot.open(f.name)
        uprootHist = uprootFile[clonedHistName]

    return (uprootHist, rootHist, expected)

@pytest.mark.parametrize("histType", [
    "uproot",
    "ROOT",
], ids = ["uproot", "ROOT"])
def testUprootHistToArray(setupHistToArray, histType):
    """ Test conversion of uproot hists to arrays. """
    (uprootHist, rootHist, expected) = setupHistToArray

    if histType == "uproot":
        inputHist = uprootHist
    elif histType == "ROOT":
        inputHist = rootHist

    h = base.Histogram.from_existing_hist(inputHist)
    np.testing.assert_allclose(h.x, expected.x)
    np.testing.assert_allclose(h.y, expected.y)
    np.testing.assert_allclose(h.errors, expected.errors)

