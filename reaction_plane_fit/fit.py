#!/usr/bine/env python

""" Defines the basic components of the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Tuple, Optional
import time

import iminuit
import numdifftools as nd
import numpy as np
import probfit

from reaction_plane_fit import base
from reaction_plane_fit import functions

logger = logging.getLogger(__name__)

# TODO: Add signal and background limit ranges to the actual fit object.
#rpf = ReactionPlaneFit(signalRegion = (0, 0.6), backgroundRegion = (0.8, 1.2))

@dataclass(frozen = True)
class FitType:
    region: str
    angle: str

class ReactionPlaneFit(ABC):
    """ Contains the reaction plane fit for one particular set of data and components.

    Attributes:
        regions (dict): Signal and background fit regions.
        components (dict): Reaction plane fit components used in the actual fit.
        resolutionParameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82"
        useLogLikelihood (bool): If true, use log likelihood cost function. Often used
            when statistics are limited. Default: False
    """
    def __init__(self, resolutionParameters: dict, useLogLikelihood: bool, signalRegion = None, backgroundRegion = None):
        self.resolutionParameters = resolutionParameters
        self.useLogLikelihood = useLogLikelihood
        self.components = {}
        self.regions = {}

        # Contains the simultaneous fit to all of the components.
        self._fit = None
        # Contains the fit results
        self.fit_result = None
        # Degress of freedom
        self.nDOF = None

    @property
    def rpAngles(self) -> list:
        """ Get the RP angles (excluding the inclusive, which is the last entry). """
        # "inclusive" must be the last entry.
        return self.angles[:-1]

    def _determine_reaction_plane_parameters(self, rpAngle) -> base.ReactionPlaneParameter:
        """ Helper to determine the reaction plane parameters. """
        return self.reactionPlaneParameters[rpAngle]

    def _validate_settings(self) -> bool:
        """ Validate the passed settings. """
        # Check that there are sufficient res params
        resParams = set(["R22", "R42", "R62", "R82"])
        goodParams = resParams.issubset(self.resolutionParameters.keys())
        if not goodParams:
            raise ValueError(f"Missing resolution parameters. Passed: {self.resolutionParameters.keys()}")

        return all([goodParams])

    def _format_input_data(self, data: dict) -> bool:
        """ Convert input data into a more convenient format.

        By using ``FitType``, we can very easily check that all fit components have the appropriate data.
        """
        # Check if it's already properly formatted.
        properlyFormatted = all(isinstance(k, FitType) for k in data.keys())
        if not properlyFormatted:
            # Convert the string keys to ``FitType`` keys.
            for region in ["signal", "background"]:
                if region in data:
                    for rp_angle in data[region]:
                        # Restruct the data within the same dict.
                        # For example, ["background"]["inPlane"] -> [FitType(region = "background", angle = "inPlane")]
                        data[FitType(region = region, angle = rp_angle)] = data[region][rp_angle]
                    # Cleanup the previous dict structure
                    del data[region]

        # Convert the data to Histogram objects.
        data = {fit_type: base.Histogram.from_existing_hist(input_hist) for fit_type, input_hist in data.items()}
        logger.debug(f"{data}")

        return data

    def _validate_data(self, data: dict) -> bool:
        """ Validate that the provided data is sufficient for the defined components.

        Args:
            data (dict): Input data with ``FitType`` keys in the corresponding data stored in the values.
        """
        # Each component must have input data.
        goodData = all(k in data.keys() for k in self.components.keys())

        return goodData

    def _determine_component_parameter_limits(self) -> dict:
        """ Determine the seed values, limits, and step sizes of parameters defined in the components.

        Note:
            These values should be specified with the proper names such that they will be recognized by Minuit.

        Args:
            None.
        Returns:
            dict: Parameter values and limits in a dictionary suitable to be used as Minuit args.
        """
        arguments = {}
        for component in self.components.values():
            arguments.update(component.determine_parameters_limits())

        return arguments

    def _run_fit(self, arguments: dict) -> Tuple[bool, iminuit.Minuit]:
        """ Make the proper calls to ``iminuit`` to run the fit.

        Args:
            arguments (dict): Arguments for the ``minuit`` object based on seed values and limits specified in
                the fit components.
        Returns:
            tuple: (fitOkay, minuit) where fitOkay (bool) is ``True`` if the fit is okay, and
                ``minuit`` (``iminuit.minuit``) is the Minuit object which was used to perform the fit.
        """
        logger.debug(f"Minuit args: {arguments}")
        minuit = iminuit.Minuit(self._fit, **arguments)

        # Perform the fit
        minuit.migrad()
        # Just in case (doesn't hurt anything, but may help in a few cases).
        minuit.hesse()
        # Plot the correlation matrix
        minuit.print_matrix()
        return (minuit.migrad_ok(), minuit)

    def fit(self, data) -> bool:
        """ Perform the actual fit.

        Args:

        Returns:
            bool: True if the fitting procedure was successful.
        """
        # Validate settings.
        goodSettings = self._validate_settings()
        if not goodSettings:
            raise ValueError("Invalid settings! Please check the inputs.")

        # Setup and validate data.
        if not data:
            raise ValueError("Must pass data to be fitted!")
        data = self._format_input_data(data)
        goodData = self._validate_data(data)
        if not goodData:
            raise ValueError(f"Insufficient data provided for the fit components. Component keys: {self.components.keys()}, Data keys: {data.keys()}")

        # Setup the fit components.
        for fitType, component in self.components.items():
            component._setup_fit(inputHist = data[fitType],
                                 resolutionParameters = self.resolutionParameters,
                                 reactionPlaneParameter = self._determine_reaction_plane_parameters(fitType.angle))

        # Extract the x locations from where the fit should be evaluated.
        print(f"{data}")
        x = next(iter(data.values())).x

        # Setup the final fit.
        self._fit = probfit.SimultaneousFit(*[component.costFunction for component in self.components.values()])
        arguments = self._determine_component_parameter_limits()

        # Perform the actual fit
        (goodFit, minuit) = self._run_fit(arguments = arguments)
        # Check if fit is considered valid
        if goodFit is False:
            raise RuntimeError("Fit is not valid!")

        # Calculate chi2/ndf (or really, min function value/ndf)
        # NDF = number of points used in the fit minus the number of free parameters.
        fixed_parameters = [k for k, v in minuit.fixed.items() if v is True]
        parameters = iminuit.util.describe(self._fit)
        free_parameters = list(set(parameters) - set(fixed_parameters))
        logger.debug(f"fixed_parameters: {fixed_parameters}, parameters: {parameters}, free_parameters: {free_parameters}")

        # Store Minuit information for calculating the errors.
        # TODO: Should this be in a separate object that can be more easily YAML storable?
        self.fit_result = base.FitResult(
            parameters = parameters,
            fixed_parameters = fixed_parameters,
            free_parameters = free_parameters,
            minimum_val = minuit.fval,
            args_at_minimum = list(minuit.args),
            values_at_minimum = dict(minuit.values),
            x = x,
            covariance_matrix = minuit.covariance,
        )
        # TODO: Store fitarg so we can recreate the minuit object?
        logger.debug(f"nDOF: {self.fit_result.nDOF}")

        # Calculate the errors.
        self.calculate_errors()

        # TODO: Store everything, including errors.

        # Return true to note success.
        return True

    def calculate_errors(self) -> np.ndarray:
        """ Calculate the errors based on values from the fit. """
        # Wrapper needed to call the function because ``numdifftools`` requires that multiple arguments
        # are in a single list. The wrapper expands that list for us
        def func_wrap(x):
            # Need to expand the arguments
            return self._fit(*x)

        # Determine the arguments for the fit function
        #argsForFuncCall = base.GetArgsForFunc(func = self._fit, xValue = None, fitContainer = fitContainer)
        #logger.debug("argsForFuncCall: {}".format(argsForFuncCall))
        args_at_minimum = self.fit_result.values_at_minimum
        logger.debug(f"args_at_minimum: {args_at_minimum}")

        # Retrieve the parameters to use in calculating the fit errors
        funcArgs = self.fit_result.free_parameters
        logger.debug(f"funcArgs: {funcArgs}")

        # Compute the derivative
        partialDerivatives = nd.Gradient(func_wrap)

        # To store the errors for each point
        # Just using "binCenters" as a proxy
        errorVals = np.zeros(len(self.fit_result.x))
        logger.debug("len(self.fit_result.x]): {}, self.fit_result.x: {}".format(len(self.fit_result.x), self.fit_result.x))
        logger.debug(f"Covariance matrix: {self.fit_result.covariance_matrix}")

        for i, val in enumerate(self.fit_result.x):
            logger.debug(f"val: {val}")
            # Add in x for the function call
            args_at_minimum["x"] = val

            #logger.debug("Actual list of args: {}".format(list(args_at_minimum.values())))

            # We need to calculate the derivative once per x value
            start = time.time()
            logger.debug(f"Calculating the gradient for point {i}.")
            partialDerivative = partialDerivatives(list(args_at_minimum.values()))
            end = time.time()
            logger.debug("Finished calculating the graident in {} seconds.".format(end - start))

            # Calculate error
            errorVal = 0
            for iName in funcArgs:
                for jName in funcArgs:
                    # Evaluate the partial derivative at a point
                    # Must be called as a list!
                    listOfArgsForFuncCall = list(args_at_minimum.values())
                    iNameIndex = listOfArgsForFuncCall.index(args_at_minimum[iName])
                    jNameIndex = listOfArgsForFuncCall.index(args_at_minimum[jName])
                    #logger.debug("Calculating error for iName: {}, iNameIndex: {} jName: {}, jNameIndex: {}".format(iName, iNameIndex, jName, jNameIndex))
                    #logger.debug("Calling partial derivative for args {}".format(argsForFuncCall))

                    # Add error to overall error value
                    errorVal += partialDerivative[iNameIndex] * partialDerivative[jNameIndex] * self.fit_result.covariance_matrix[(iName, jName)]

            # Modify from error squared to error
            errorVal = np.sqrt(errorVal)

            # Store
            #logger.debug("i: {}, errorVal: {}".format(i, errorVal))
            errorVals[i] = errorVal

        return errorVals

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
    def determine_fit_function(self, resolutionParameters: dict, reactionPlaneParameter: base.ReactionPlaneParameter) -> None:
        """ Use the class parameters to determine the fit function and store it. """

    def _setup_fit(self, inputHist: base.Histogram, resolutionParameters: dict, reactionPlaneParameter: base.ReactionPlaneParameter) -> None:
        """ Setup the fit using information from the input hist.

        Args:
            inputHist (ROOT.TH1): The histogram to be fit by this function.
            resolutionParameters (dict): Maps resolution parameters of the form "R22" (for
                the R_{2,2} parameter) to the value. Expects "R22" - "R82"
            reactionPlaneParameter (base.ReactionPlaneParameter): Reaction plane parameters for the selected
                reaction plane.
        Returns:
            None: The fit object is fully setup.
        """
        # Setup the fit itself.
        self.determine_fit_function(resolutionParameters = resolutionParameters,
                                    reactionPlaneParameter = reactionPlaneParameter)

        #hist = base.Histogram.from_existing_hist(hist = inputHist)
        limitedHist = self.set_data_limits(hist = inputHist)
        # TODO: Scale histogram by nEvents if necessary

        # Determine the cost function
        self.costFunction = self.cost_function(hist = limitedHist)

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
            logger.debug(f"Using log likelihood for {self.fitType}, {self.rpOrientation}, {self.region}")
            # Generally will use when statistics are limited.
            # Errors are extracted by assuming a Poisson distribution, so we don't need to pass them explicitly (?)
            costFunction = probfit.UnbinnedLH(f = self.fitFunction,
                                              data = hist.x)
        else:
            logger.debug(f"Using Chi2 for {self.fitType}, {self.rpOrientation}, {self.region}")
            costFunction = probfit.Chi2Regression(f = self.fitFunction,
                                                  x = hist.x,
                                                  y = hist.y,
                                                  error = hist.errors)

        # Return the function so that it can be stored. We explicitly return it
        # so that it is clear that it is being assigned.
        return costFunction

    def determine_parameters_limits(self) -> dict:
        """ Determine the parameter seed values, limits, step sizes for the component.

        Note:
            These values should be specified with the proper names such that they will be recognized by Minuit.

        Args:
            None
        Returns:
            Dictionary of the function arguments which specifies their
        """
        arguments = {}
        if self.region == "signal":
            # Signal default parameters
            # NOTE: The error should be approximately 10% of the value to ensure that the step size of the fit
            #       is correct.
            nsSigmaInit = 0.07
            asSigmaInit = 0.2
            sigmaLowerLimit = 0.025
            sigmaUpperLimit = 0.35
            signalLimits = {
                "nsSigma": nsSigmaInit,
                "limit_nsSigma": (sigmaLowerLimit, sigmaUpperLimit), "error_nsSigma": 0.1 * nsSigmaInit,
                "asSigma": asSigmaInit,
                "limit_asSigma": (sigmaLowerLimit, sigmaUpperLimit), "error_asSigma": 0.1 * asSigmaInit,
                "signalPedestal": 0.0, "fix_signalPedestal": True,
            }

            if self.rpOrientation != "inclusive":
                # Add the reaction plane prefix so the arguments don't conflict.
                signalLimits = iminuit.util.fitarg_rename(signalLimits, lambda name: self.rpOrientation + "_" + name)

            # Add in the signal limits regardless of RP orientation
            arguments.update(signalLimits)

        # Background arguments
        # These should consistently be defined regardless of the fit definition.
        # Note that if the arguments already exist, they made be updated to the same value. However, this shouldn't
        # cause any problems of have any effect.
        # NOTE: The error should be approximately 10% of the value to ensure that the step size of the fit
        #       is correct.
        backgroundLimits = {
            "v2_t": 0.02, "limit_v2_t": (0, 0.50), "error_v2_t": 0.001,
            "v2_a": 0.02, "limit_v2_a": (0, 0.50), "error_v2_a": 0.001,
            "v4_t": 0.01, "limit_v4_t": (0, 0.50), "error_v4_t": 0.001,
            "v4_a": 0.01, "limit_v4_a": (0, 0.50), "error_v4_a": 0.001,
            "v3": 0.0, "limit_v3": (-0.1, 0.5), "error_v3": 0.001,
            "v1": 0.0, "fix_v1": True,
        }
        arguments.update(backgroundLimits)

        return arguments

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

    def determine_fit_function(self, resolutionParameters: dict, reactionPlaneParameter: base.ReactionPlaneParameter) -> None:
        self.fitFunction = functions.determine_signal_dominated_fit_function(
            rpOrientation = self.rpOrientation,
            resolutionParameters = resolutionParameters,
            reactionPlaneParameter = reactionPlaneParameter
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

    def determine_fit_function(self, resolutionParameters: dict, reactionPlaneParameter: base.ReactionPlaneParameter) -> None:
        self.fitFunction = functions.determine_background_fit_function(
            rpOrientation = self.rpOrientation,
            resolutionParameters = resolutionParameters,
            reactionPlaneParameter = reactionPlaneParameter,
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
