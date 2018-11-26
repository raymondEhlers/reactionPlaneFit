#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

@dataclass
class FitResult:
    """ Store the Fit Result.

    Note:
        free_parameters + fixed_parameters = parameters

    Attributes:
        parameters (list): Names of the parameters used in the fit.
        free_parameters (list): Names of the free parameters used in the fit.
        fixed_parameters (list): Names of the fixed parameters used in the fit.
        minimul_val (float): Minimum value of the fit when it coverages. This is the chi2 value for a
            chi2 minimization fit.
        nDOF (int): Number of degrees of freedom.
        args_at_minimum (list): Numerical values of parameters at the convergence.
        values_at_minimum (dict): Keys are the names of parameters, while values are the numerical values
            at convergence. This is somewhat redundant with ``args_at_minimum``, but both can be useful.
        x (list): x values where the fit result should be evaluated.
        covariance_matrix (dict): Keys are tuples with (paramNameA, paramNameB), and the values are covariance
            between the specified parameters. Note that fixed parameters are _not_ included in this matrix.
    """
    # TODO: Fold into main fit object?
    parameters: list
    free_parameters: list
    fixed_parameters: list
    minimum_val: float
    args_at_minimum: list
    values_at_minimum: dict
    x: list
    covariance_matrix: dict

    @property
    def nDOF(self):
        return len(self.x) - len(self.free_parameters)

@dataclass
class ReactionPlaneParameter:
    """ Parameters that defined a reaction plane.

    Attributes:
        orientation (str): Reaction plane orientation.
        phiS (float): Center of the reaction plane bin.
        c (float): Width of the reaction plane bin.
    """
    orientation: str
    phiS: float
    c: float

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
        logger.debug(f"{hist}, {type(hist)}")
        logger.debug(f"hasattr: {hasattr(hist, 'values')}")
        if hasattr(hist, "values"):
            (x, y, errors) = cls._from_uproot(hist)
        else:
            # Handle traditional ROOT hists
            (x, y, errors) = cls._from_th1(hist)

        return cls(x = x, y = y, errors = errors)

