#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class Histogram:
    """ Contains histogram data.

    Attributes:
        x (np.ndarray): The bin centers.
        y (np.ndarray): The bin value.
        errors (np.ndarray): The bin errors.
    """
    x: np.ndarray
    y: np.ndarray
    errors: np.ndarray

    @staticmethod
    def _from_uproot(hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Convert a uproot histogram to a set of array for creating a Histogram.

        Note:
            Underflow and overflow bins are excluded! Bins are assumed to be fixed
            size.

        Args:
            hist (uproot.hist.TH1*): Input histogram.
        Returns:
            tuple: (x, y, errors) where x is the bin centers, y is the bin values, and
                errors are the bin errors.
        """
        # This excluces underflow and overflow
        (y, edges) = hist.numpy()

        # Assume uniform bin size
        binSize = (hist.high - hist.low) / hist.numbins
        # Shift all of the edges to the center of the bins
        # (and drop the last value, which is now invalid)
        x = edges[:-1] + binSize / 2.0

        # Also retrieve errors from sumw2.
        # If more sophistication is needed, we can modify this to follow the approach to
        # calculating bin errors from TH1::GetBinError()
        errors = np.sqrt(hist.variances)

        return (x, y, errors)

    @staticmethod
    def _from_th1(hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Convert a TH1 histogram to a Histogram.

        Note:
            Underflow and overflow bins are excluded! Bins are assumed to be fixed
            size.

        Args:
            hist (ROOT.TH1):
        Returns:
            tuple: (x, y, errors) where x is the bin centers, y is the bin values, and
                errors are the bin errors.
        """
        # Handle traditional ROOT hists
        xAxis = hist.GetXaxis()
        # Don't include overflow
        xBins = range(1, xAxis.GetNbins() + 1)
        # NOTE: The bin error is stored with the hist, not the axis.
        x = np.array([xAxis.GetBinCenter(i) for i in xBins])
        y = np.array([hist.GetBinContent(i) for i in xBins])
        errors = np.array([hist.GetBinError(i) for i in xBins])

        return (x, y, errors)

    @classmethod
    def from_existing_hist(cls, hist):
        """ Convert an existing histogram.

        Note:
            Underflow and overflow bins are excluded! Bins are assumed to be fixed
            size.

        Args:
            hist (uproot.rootio.TH1* or ROOT.TH1):
        Returns:
            Histogram: Data class with x, y, and errors
        """
        # "values" is a proxy for if we have an uproot hist.
        if hasattr(hist, "values"):
            (x, y, errors) = cls._from_uproot(hist)
        else:
            # Handle traditional ROOT hists
            (x, y, errors) = cls._from_th1(hist)

        return cls(x = x, y = y, errors = errors)

