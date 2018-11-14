#!/usr/bine/env python

""" Defines the basic components of the reaction plane fit.

"""

from abc import ABC, abstractmethod
from dataclass import dataclass
from typing import Optional
import logging

import numpy as np
import probfit

logger = logging.getLogger(__name__)

from reactionPlaneFit import base

#rpf = ReactionPlaneFit(signalRegion = (0, 0.6), backgroundRegion = (0.8, 1.2))

class ReactionPlaneFit(object):
    """ Contains the reaction plane fit for one particular set of data and components.

    Attributes:
        type (tuple): Fit type.
        regions (dict): Signal and background fit regions.
        components (dict): Reaction plane fit components used in the actual fit.
        resolutionParameters (dict): Event plane resolution parameters for the fit.
    """
    def __init__(self, type = None, signalRegion = None, backgroundRegion = None):
        self.type = type
        self.components = {}
        self.regions = {}
        self.resolutionParameters = {}

        # Points where the fit will be evaluated.
        self.x = np.array([])

        if signalRegion:
            self.regions["signal"] = signalRegion
        if backgroundRegion:
            self.regions["backgroundRegion"] = backgroundRegion

    def _add_component(self, component):
        #self.components[] = component
        pass

    def _validate_fit_parameters(self):
        """ Check that all expected fit components are available.

        """
        pass

    def fit(self, data):
        """

        """
        # Setup the fit components.
        for component in self.components:
            component.setup_fit(...)

        # Extract the x locations from where the fit should be evaluated.

        # Perform the actual fit.

        # Calculate the errors.

        # Store everything.

@dataclass
class FitType:
    region: str
    angle: str

class FitComponent(ABC):
    """ A component of the fit.

    Args:
        region (str): Region in which the fit component is applied.
        rpAngle (str): The reaction plane orientation of the fit.
        resolutionParameters (dict): Maps resolution paramaeters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82"
        useLogLikelihood (bool): If true, use log likelihood cost function. Often used
            when statistics are limited. Default: False

    Attributes:
        fitType (FitType): Overall type of the fit.
    """
    def __init__(self, region: str, rpAngle: str, hist: base.Histogram, resolutionParameters: dict, useLogLikelihood: bool = False) -> None:
        self.set_fit_type(region = region, angle = rpAngle)
        self.resolutionParameters = resolutionParameters
        self.useLogLikelihood = useLogLikelihood

        # Additional values
        # Fit cost function
        self.costFunction = None

        # Will be determine in each subclass
        # Called last to ensure that all variables are available.
        self.fitFunction = None
        self.determine_fit_function()

    def set_fit_type(self, region: Optional[str] = None, angle: Optional[str] = None):
        if not region:
            region = self.fitType.region
        if not angle:
            angle = self.fitType.angle
        self.fitType = FitType(region = region, angle = angle)

    @property
    def rpOrientation(self) -> str:
        return self.fitType.angle

    @property
    def region(self) -> str:
        return self.fitType.region

    @abstractmethod
    def determine_fit_function(self):
        """ Use the class parameters to determine the fit function and store it. """

    def setup_fit(self, inputHist):
        """ Setup the fit using information from the input hist.

        Args:
            inputHist (ROOT.TH1): The histogram to be fit by this function.
        Returns:
            None: The fit object is fully setup.
        """
        hist = base.Histogram.from_existing_hist(hist = inputHist)
        limitedHist = self.set_data_limits(hist = hist)
        # Detemrine the cost function
        self.costFunction = self.cost_function(hist = limitedHist)

    def set_data_limits(self, hist):
        """ Extract the data from the histogram. """
        return hist

    def cost_function(self, hist: Optional[base.Histogram] = None, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, errors: Optional[np.ndarray] = None):
        """ Define the cost function.

        Called when setting up a fit object.

        Note:
            We don't want to use the binned cost function versions - they will bin
            the data that is given, which is definitely not what we want. instead,
            use the unbinned functions (as defined by probfit).

        Args:
            x (np.ndarray): The x values associated with an input histogram.
            y (np.ndarray): The y values associated with an input histogram.
            errors (np.ndarray): The errors associated with an input histogram.
        Returns:
            probfit.costFunc: The defined cost function.
        """
        # Argument validation and properly format individual array is necessary.
        if not hist:
            # Validate the individual arguments
            if not x or not y or not errors:
                raise ValueError("Must provide x, y, and errors arrays.")
            # Then format them so they can be used.
            hist = base.Histogram(x = x, y = y, errors = errors)
        else:
            # These shouldn't be set if we're using a histogram.
            if x or y or errors:
                raise ValueError("Provided histogram, and x, y, or errors. Must provide only the histogram!")

        if self.useLogLikelihood:
            logger.debug(f"Using log likelihood for {self.fitType}, {self.rp_orientation}, {self.region}")
            # Generally will use when statistics are limited.
            # Errors are extracted by assuming a poisson distribution, so we don't need to pass them explcitily (?)
            costFunction = probfit.UnbinnedLH(f = self.fitFunction,
                                              data = hist.x)
        else:
            logger.debug(f"Using Chi2 for {self.fitType}, {self.rp_orientation}, {self.region}")
            costFunction = probfit.Chi2Regression(f = self.fitFunction,
                                                  x = hist.x,
                                                  y = hist.y,
                                                  error = hist.errors)

        # Return the function so that it can be stored. We explicitly return it
        # so that it is clear that it is being assigned.
        return costFunction

class SignalFitComponent(FitComponent):
    """ Fit component in the signal region.

    Args:

    """
    def __init__(self, *args, **kwargs):
        super().__init__(**args, **kwargs)

        self.set_fit_type(region = "signal")
        self.determine_fit_function()

    def determine_fit_function(self):
        self.fitFunction = determineSignalDominatedFitFunction(rpOrientation = self.rpOrientation,
                                                               resolutionParameters = self.resolutionParameters)

class BackgroundFitComponent(FitComponent):
    """ Fit component in the background region.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolutionParameters = {}

        self.set_fit_type(region = "signal")
        self.determine_fit_function()

    def determine_fit_function(self):
        self.fitFunction = determineBackgroundFitFunction(rpOrientation = self.rpOrientation,
                                                          resolutionParameters = self.resolutionParameters)

    def set_data_limits(self, hist):
        """ Set the limits of the fit to only use near-side data (ie dPhi < pi/2)

        Only the near-side will be used for the fit to avoid the signal at large
        dPhi on the away-side.

        Args:
            hist (base.Histogram): The input data.
        Returns:
            base.Histogram: The data limited to the near-side.
        """
        # Use only near-side data (ie dPhi < pi/2)
        NSrange = int(len(hist.x) / 2.0)
        x = hist.x[:NSrange]
        y = hist.y[:NSrange]
        errors = hist.errors[:NSrange]

        # Return a data limits histogram
        return base.Histogram(x = x, y = y, errors = errors)
