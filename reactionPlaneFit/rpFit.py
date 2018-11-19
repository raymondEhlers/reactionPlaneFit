#!/usr/bine/env python

""" Defines the basic components of the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional
import time

import iminuit
import numdifftools as nd
import numpy as np
import probfit

from reactionPlaneFit import base
from reactionPlaneFit import fitFunctions

logger = logging.getLogger(__name__)

# TODO: Add signal and background limit ranges to the actual fit object.
#rpf = ReactionPlaneFit(signalRegion = (0, 0.6), backgroundRegion = (0.8, 1.2))

@dataclass
class ReactionPlaneParameter:
    angle: str
    phiS: float
    c: float

class ReactionPlaneFit(ABC):
    """ Contains the reaction plane fit for one particular set of data and components.

    Attributes:
        type (tuple): Fit type.
        regions (dict): Signal and background fit regions.
        components (dict): Reaction plane fit components used in the actual fit.
        resolutionParameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82"
    """
    def __init__(self, type, resolutionParameters: dict, signalRegion = None, backgroundRegion = None):
        self.type = type
        self.resolutionParameters = resolutionParameters
        self.components = {}
        self.regions = {}

        # Points where the fit will be evaluated.
        self.x = np.array([])

    @property
    def rpAngles(self):
        """ Get the RP angles (excluding the inclusive). """
        # "all" is the last entry.
        return self.angles[:-1]

    def _validate_fit_data(self):
        """ Check that all expected fit components are available.

        """
        pass

    def determine_reaction_plane_parameters(self, rpAngle) -> ReactionPlaneParameter:
        #return self.phiS[rpAngle], self.c[rpAngle]
        return self.reactionPlaneParameters[rpAngle]

    def _format_input_data(self, data):
        """ Convert input data into a more convenient format.

        By using ``FitType``, we can very easily check that all fit components have the appropriate data.
        """
        # Check if it's already properly formatted.
        properlyFormatted = all(isinstance(FitType, k) for k in data.keys())
        if properlyFormatted:
            return data

        # If not, convert the string keys to ``FitType`` keys.
        returnData = {}
        for region in ["signal", "background"]:
            if region in data:
                for rpAngle in data[region]:
                    returnData[FitType(region = region, angle = rpAngle)] = data[region][rpAngle]
        return returnData

    def _validate_data(self, data: dict) -> bool:
        """ Validate that the provided data is sufficient for the defined components.

        Args:
            data (dict): Input data with ``FitType`` keys in the corresponding data stored in the values.
        """
        # Each component must have input data.
        goodData = all(k in data.keys() for k in self.components.keys())

        return goodData

    def defineArgs(self):
        """

        Returns:
            dict: Arguments for the fit.
        """
        minuitArgs = {}

        # Get args from each component.
        # Will need some strategy to merge them.

        # Background arguments
        # These should consistent regardless of the fit definition.
        # NOTE: The error should be approximately 10% of the value to ensure that the step size of the fit is correct.
        backgroundLimits = dict(
            v2_t = 0.02, limit_v2_t = (0, 0.50), error_v2_t = 0.001,
            v2_a = 0.02, limit_v2_a = (0, 0.50), error_v2_a = 0.001,
            v4_t = 0.01, limit_v4_t = (0, 0.50), error_v4_t = 0.001,
            v4_a = 0.01, limit_v4_a = (0, 0.50), error_v4_a = 0.001,
            v3 = 0.0, limit_v3 = (-0.1, 0.5), error_v3 = 0.001,
            v1 = 0.0, fix_v1 = True
        )
        minuitArgs.update(backgroundLimits)

        return minuitArgs

    def fit(self, data):
        """ Perform the actual fit.

        """
        # Setup data.
        data = self._format_data(data)
        goodData = self._validate_data(data)

        if not goodData:
            raise ValueError("Insufficient data provided for the fit components. Component keys: {self.components.keys()}, Data keys: {data.keys()}")

        # Setup the fit components.
        for fitType, component in self.components.items():
            component.setup_fit(inputHist = data[fitType.angle],
                                resolutionParameters = self.resolutionParameters,
                                reactionPlaneParameters = self.determine_reaction_plane_parameters(fitType.angle))

        # Extract the x locations from where the fit should be evaluated.
        x = next(data.values()).x

        # Setup the final fit.
        self._fit = probfit.SimultaneousFit(*[component.costFunction for component in self.fitComponents.values()])
        args = self.defineArgs()

        # Perform the actual fit
        logger.debug(f"Minuit args: {args}")
        minuit = iminuit.Minuit(self._fit, **args)

        # Perform the fit
        minuit.migrad()
        # Just in case (doesn't hurt anything, but may help in a few cases).
        minuit.hesse()
        # Plot the correlation matrix
        minuit.print_matrix()
        # Check if fit is considered valid
        if not minuit.migrad_ok():
            raise RuntimeError("Fit is not valid!")

        # Store minuit information.

        # Calculate the errors.
        self.calculate_errors()

        # Store everything.

    def calculate_errors(self):
        """ Calculate the errors based on values from the fit. """
        # Wrapper needed to call the function because ``numdifftools`` requires that multiple arguments
        # are in a single list. The wrapper expands that list for us
        def func_wrap(x):
            # Need to expand the arguments
            return self.fitFunction(*x)

        # Determine the arguments for the fit function
        argsForFuncCall = base.GetArgsForFunc(func = self.fitFunction, xValue = None, fitContainer = fitContainer)
        logger.debug("argsForFuncCall: {}".format(argsForFuncCall))

        # Retrieve the parameters to use in calculating the fit errors
        funcArgs = probfit.describe(self.fitFunction)
        # Remove "x" as an argument, because we don't want to evaluate the error on it
        funcArgs.pop(funcArgs.index("x"))
        # Remove free parameters, as they won't contribute to the error and will cause problems for the gradient
        for param in fitContainer.params:
            if "fix_" in param and fitContainer.params[param] is True:
                # This parameter is fixed. We need to remove it from the funcArgs!
                funcArgParamName = param.replace("fix_", "")
                # Remove it from funcArgs if it exists
                if funcArgParamName in funcArgs:
                    funcArgs.pop(funcArgs.index(funcArgParamName))
        logger.debug("funcArgs: {}".format(funcArgs))

        # Compute the derivative
        partialDerivatives = nd.Gradient(func_wrap)

        # To store the errors for each point
        # Just using "binCenters" as a proxy
        errorVals = np.zeros(len(self.x))
        #logger.debug("len(self.x]): {}, self.x: {}".format(len(self.x), self.x))

        for i, val in enumerate(self.x):
            # Add in x for the function call
            argsForFuncCall["x"] = val

            #logger.debug("Actual list of args: {}".format(list(argsForFuncCall.itervalues())))

            # We need to calculate the derivative once per x value
            start = time.time()
            logger.debug("Calculating the gradient for point {}.".format(i))
            partialDerivative = partialDerivatives(list(argsForFuncCall.itervalues()))
            end = time.time()
            logger.debug("Finished calculating the graident in {} seconds.".format(end - start))

            # Calculate error
            errorVal = 0
            for iName in funcArgs:
                for jName in funcArgs:
                    # Evaluate the partial derivative at a point
                    # Must be called as a list!
                    listOfArgsForFuncCall = list(argsForFuncCall.itervalues())
                    iNameIndex = listOfArgsForFuncCall.index(argsForFuncCall[iName])
                    jNameIndex = listOfArgsForFuncCall.index(argsForFuncCall[jName])
                    #logger.debug("Calculating error for iName: {}, iNameIndex: {} jName: {}, jNameIndex: {}".format(iName, iNameIndex, jName, jNameIndex))
                    #logger.debug("Calling partial derivative for args {}".format(argsForFuncCall))

                    # Add error to overall error value
                    errorVal += partialDerivative[iNameIndex] * partialDerivative[jNameIndex] * fitContainer.covarianceMatrix[(iName, jName)]

            # Modify from error squared to error
            errorVal = np.sqrt(errorVal)

            # Store
            #logger.debug("i: {}, errorVal: {}".format(i, errorVal))
            errorVals[i] = errorVal

        return errorVals

class ReactionPlaneFit3Angles(ReactionPlaneFit):
    """ Reaction plane fit for 3 reaction plane angles.

    """
    angles = ["inPlane", "midPlane", "outOfPlane", "all"]
    reactionPlaneParameters = {
        "inPlane": ReactionPlaneParameter(angle = "inPlane",
                                          phiS = 0,
                                          c = np.pi / 6.),
        # NOTE: This c value is halved in the fit to account for the four non-continuous regions
        "midPlane": ReactionPlaneParameter(angle = "midPlane",
                                           phiS = np.pi / 4.,
                                           c = np.pi / 12.),
        "outOfPlane": ReactionPlaneParameter(angle = "outOfPlane",
                                             phiS = np.pi / 2.,
                                             c = np.pi / 6.),
    }

@dataclass
class FitType:
    region: str
    angle: str

class ReactionPlane3AngleBackgroundFit(ReactionPlaneFit3Angles):
    """ RPF for background region in 3 reaction plane angles.

    This is a simple helper class to define the necessary fit component. Contains fit compnents for
    3 background RP angles.

    Args:
        Same as for ``ReactionPlaneFit``.
    """
    def __init__(self, *args, **kwargs):
        # Create the base class first
        super().__init__(*args, **kwargs)

        # Setup the fit components
        for angle in self.angles:
            fitType = FitType(region = "background", angle = angle)
            self.component[fitType] = BackgroundFitComponent(fitType = fitType,
                                                             resolutionParameters = self.resolutionParameters,
                                                             useLogLikelihood = self.useLogLikelihood)

class ReactionPlane3AngleInclusiveSignalFit(ReactionPlaneFit3Angles):
    """ RPF for inclusive signal region, and background region in 3 reaction planes angles.

    This is a simple helper class to define the necessary fit component. Contains an inclusive signal fit,
    and 3 background RP angles.

    Args:
        Same as for ``ReactionPlaneFit``.
    """
    def __init__(self, *args, **kwargs):
        # Create the base class first
        super().__init__(*args, **kwargs)

        # Setup the fit components
        fitType = FitType(region = "background", angle = "inclusive")
        self.component[fitType] = SignalFitComponent(fitType = fitType,
                                                     resolutionParameters = self.resolutionParameters,
                                                     useLogLikelihood = self.useLogLikelihood)
        for angle in self.angles:
            fitType = FitType(region = "background", angle = angle)
            self.component[fitType] = BackgroundFitComponent(fitType = fitType,
                                                             resolutionParameters = self.resolutionParameters,
                                                             useLogLikelihood = self.useLogLikelihood)


class ReactionPlane3AngleSignalFit(ReactionPlaneFit3Angles):
    """ RPF for signal and background regions with 3 reaction plane angles.

    This is a simple helper class to define the necessary fit component.  Contains 3 signal
    angles and 3 background RP angles.

    Args:
        Same as for ``ReactionPlaneFit``.
    """
    def __init__(self, *args, **kwargs):
        # Create the base class first
        super().__init__(*args, **kwargs)

        # Setup the fit components
        for region, fitComponent in [("signal", SignalFitComponent), ("background", BackgroundFitComponent)]:
            for angle in self.angles:
                fitType = FitType(region = region, angle = angle)
                self.component[fitType] = fitComponent(fitType = fitType,
                                                       resolutionParameters = self.resolutionParameters,
                                                       useLogLikelihood = self.useLogLikelihood)

class FitComponent(ABC):
    """ A component of the fit.

    Args:
        region (str): Region in which the fit component is applied.
        rpAngle (str): The reaction plane orientation of the fit.
        resolutionParameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82"
        useLogLikelihood (bool): If true, use log likelihood cost function. Often used
            when statistics are limited. Default: False

    Attributes:
        fitType (FitType): Overall type of the fit.
        useLogLikelihood (bool): True if the fit should be performed using log likelihood.
        fitFunction (function): Function of the component.
        costFunction (probfit.costFunc): Cost function associated with the fit component.
    """
    def __init__(self, fitType: FitType, resolutionParameters: dict, useLogLikelihood: bool = False) -> None:
        self.fitType = fitType
        self.useLogLikelihood = useLogLikelihood

        # Additional values
        # Will be determine in each subclass
        # Called last to ensure that all variables are available.
        self.fitFunction = None
        # Fit cost function
        self.costFunction = None

    def set_fit_type(self, region: Optional[str] = None, angle: Optional[str] = None):
        """ Update the fit type.

        Will use the any of the existing parameters if they are not specified.

        Args:
            region (str): Region of the fit type.
            angle (str): Angle of the fit type.
        Returns:
            None: The fit type is set in the component.
        """
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
    def determine_fit_function(self, resolutionParameters: dict, reactionPlaneParameters: ReactionPlaneParameter) -> None:
        """ Use the class parameters to determine the fit function and store it. """

    def setup_fit(self, inputHist: base.Histogram, resolutionParameters: dict, reactionPlaneParameters: ReactionPlaneParameter) -> base.Histogram:
        """ Setup the fit using information from the input hist.

        Args:
            inputHist (ROOT.TH1): The histogram to be fit by this function.
            resolutionParameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82"
        Returns:
            None: The fit object is fully setup.
        """
        # Setup the fit itself.
        self.determine_fit_function(resolutionParameters = resolutionParameters,
                                    reactionPlaneParameters = reactionPlaneParameters)

        hist = base.Histogram.from_existing_hist(hist = inputHist)
        limitedHist = self.set_data_limits(hist = hist)
        # Determine the cost function
        self.costFunction = self.cost_function(hist = limitedHist)

        # TODO: Scale histogram by nEvents if necessary

    def set_data_limits(self, hist: base.Histogram) -> base.Histogram:
        """ Extract the data from the histogram. """
        return hist

    def cost_function(self, hist: Optional[base.Histogram] = None, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, errors: Optional[np.ndarray] = None):
        """ Define the cost function.

        Called when setting up a fit object.

        Note:
            We don't want to use the binned cost function versions - they will bin
            the data that is given, which is definitely not what we want. Instead,
            use the unbinned functions (as defined by ``probfit``).

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
            # Errors are extracted by assuming a Poisson distribution, so we don't need to pass them explicitly (?)
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
        rpAngle (str): Reaction plane angle for the component.
        *args (list): Only use named args.
        **kwargs (dict): Named arguments to be passed on to the base component.
    """
    def __init__(self, rpAngle, *args, **kwargs):
        # Validation
        if args:
            raise ValueError(f"Please specify all variables by name. Gave positional arguments: {args}")
        super().__init__(FitType(region = "signal", angle = rpAngle), **kwargs)

    def determine_fit_function(self, resolutionParameters: dict, reactionPlaneParameters: ReactionPlaneParameter) -> None:
        self.fitFunction = fitFunctions.determine_signal_dominated_fit_function(
            rpOrientation = self.rpOrientation,
            resolutionParameters = resolutionParameters,
            reactionPlaneParameters = reactionPlaneParameters
        )

class BackgroundFitComponent(FitComponent):
    """ Fit component in the background region.

    Args:
        rpAngle (str): Reaction plane angle for the component.
        *args (list): Only use named args.
        **kwargs (dict): Named arguments to be passed on to the base component.
    """
    def __init__(self, rpAngle, *args, **kwargs):
        # Validation
        if args:
            raise ValueError(f"Please specify all variables by name. Gave positional arguments: {args}")
        super().__init__(FitType(region = "background", angle = rpAngle), **kwargs)

    def determine_fit_function(self, resolutionParameters: dict, reactionPlaneParameters: ReactionPlaneParameter) -> None:
        self.fitFunction = fitFunctions.determine_background_fit_function(
            rpOrientation = self.rpOrientation,
            resolutionParameters = resolutionParameters,
            reactionPlaneParameters = reactionPlaneParameters
        )

    def set_data_limits(self, hist: base.Histogram) -> base.Histogram:
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
