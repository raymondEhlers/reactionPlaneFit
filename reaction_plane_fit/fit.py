#!/usr/bine/env python

""" Defines the basic components of the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC, abstractmethod
import logging
from typing import cast, Callable, Dict, Optional, Tuple, Union
import time

import iminuit
import numdifftools as nd
import numpy as np
from pachyderm import histogram
from pachyderm.typing_helpers import Hist
from pachyderm import yaml
import probfit

from reaction_plane_fit import base
from reaction_plane_fit import functions

logger = logging.getLogger(__name__)

# Type helpers
InputData = Dict[str, Dict[str, Union[Hist, histogram.Histogram1D]]]
Data = Dict[base.FitType, histogram.Histogram1D]
FitArguments = Dict[str, Union[bool, float, Tuple[float, float], Tuple[int, int]]]
ResolutionParameters = Dict[str, float]

class ReactionPlaneFit(ABC):
    """ Contains the reaction plane fit for one particular set of data and fit components.

    Args:
        resolution_parameters (dict): Maps resolution parameters of the form "R22" (for the R_{2,2} parameter)
            to the value. Expects "R22" - "R82" (even RP only).
        use_log_likelihood (bool): If true, use log likelihood cost function. Often used when statistics are
            limited. Default: False
        signal_region: Min and max extraction range for the signal dominated region. Should be
            provided as the absolute value (ex: (0., 0.6)).
        background_region: Min and max extraction range for the background dominated region. Should
            be provided as the absolute value (ex: (0.8, 1.2)).
        use_minos: Calculate the errors using Minos in addition to Hesse.

    Attributes:
        resolution_parameters (dict): Maps resolution parameters of the form "R22" (for the R_{2,2} parameter)
            to the value. Expects "R22" - "R82" (even RP only).
        use_log_likelihood (bool): If true, use log likelihood cost function. Often used when statistics are
            limited. Default: False
        signal_region: Min and max extraction range for the signal dominated region. Should be
            provided as the absolute value (ex: (0., 0.6)).
        background_region: Min and max extraction range for the background dominated region. Should
            be provided as the absolute value (ex: (0.8, 1.2)).
        use_minos: Calculate the errors using Minos in addition to Hesse. It will give a better error estimate, but
            it will be much slower. Errors will be printed but not stored because we don't store the errors directly
            (rather, we store the covariance matrix, which one cannot extract from Minos). By printing out the values,
            the use can check that the Hesse and Minos errors are similar, which will indicate that the function is
            well approximated by a hyperparabola at the minima (and thus Hesse errors are okay to be used).
        fit_result (base.RPFitResult): Result of the RP fit.
        _fit (Callable): Fit function for the RP fit.
    """
    # RP orientations (including inclusive). Should be overridden by the derived class.
    _rp_orientations: list = []
    reaction_plane_parameters: Dict[str, base.ReactionPlaneParameter] = {}

    def __init__(self, resolution_parameters: ResolutionParameters,
                 use_log_likelihood: bool,
                 signal_region: Tuple[float, float] = None,
                 background_region: Tuple[float, float] = None,
                 use_minos: bool = False):
        self.resolution_parameters = resolution_parameters
        self.use_log_likelihood = use_log_likelihood
        self.components: Dict[base.FitType, FitComponent] = {}
        self.regions = {"signal": signal_region, "background": background_region}
        self.use_minos = use_minos

        # Contains the simultaneous fit to all of the components.
        self._fit = Callable[..., float]
        # Contains the fit results
        self.fit_result: base.RPFitResult

    @property
    def rp_orientations(self) -> list:
        """ Get the RP orientations (excluding the inclusive, which is the last entry). """
        # "inclusive" must be the last entry.
        return self._rp_orientations[:-1]

    def _determine_reaction_plane_parameters(self, rp_orientation) -> base.ReactionPlaneParameter:
        """ Helper to determine the reaction plane parameters. """
        return self.reaction_plane_parameters[rp_orientation]

    def _setup_component_fit_functions(self) -> bool:
        """ Setup component fit functions.

        This should be called in the constructor of derived objects.

        Args:
            None.
        Returns:
            True if the component fit functions were setup successfully.
        """
        for fit_type, component in self.components.items():
            component.determine_fit_function(
                resolution_parameters = self.resolution_parameters,
                reaction_plane_parameter = self._determine_reaction_plane_parameters(fit_type.orientation)
            )

        return True

    def _validate_settings(self) -> bool:
        """ Validate the passed settings. """
        # Check that there are sufficient res params
        resParams = set(["R22", "R42", "R62", "R82"])
        good_params = resParams.issubset(self.resolution_parameters.keys())
        if not good_params:
            raise ValueError(f"Missing resolution parameters. Passed: {self.resolution_parameters.keys()}")

        return all([good_params])

    def _format_input_data(self, data: Union[InputData, Data]) -> Data:
        """ Convert input data into a more convenient format.

        By using ``FitType``, we can very easily check that all fit components have the appropriate data.

        Args:
            data: Input data to be formatted.
        Returns:
            Properly formatted data, with ``FitType`` keys and histograms as values.
        """
        # Check if it's already properly formatted.
        formatted_data: Data = {}
        properly_formatted = all(isinstance(k, base.FitType) for k in data.keys())
        if not properly_formatted:
            # Convert the string keys to ``FitType`` keys.
            # Help out mypy
            data = cast(InputData, data)
            for region in ["signal", "background"]:
                if region in data:
                    for rp_orientation in data[region]:
                        # Convert the key for storing the data.
                        # For example, ["background"]["in_plane"] -> [FitType(region = "background",
                        # orientation = "in_plane")]
                        hist = data[region][rp_orientation]
                        formatted_data[base.FitType(region = region, orientation = rp_orientation)] = hist
        else:
            # Help out mypy
            data = cast(Data, data)
            formatted_data = data

        # Convert the data to Histogram objects.
        formatted_data = {
            fit_type: histogram.Histogram1D.from_existing_hist(input_hist)
            for fit_type, input_hist in formatted_data.items()
        }
        logger.debug(f"{formatted_data}")

        return formatted_data

    def _validate_data(self, data: dict) -> bool:
        """ Validate that the provided data is sufficient for the defined components.

        Args:
            data (dict): Input data with ``FitType`` keys in the corresponding data stored in the values.
        """
        # Each component must have input data.
        good_data = all(k in data.keys() for k in self.components.keys())

        return good_data

    def _determine_component_parameter_limits(self, user_arguments: FitArguments) -> FitArguments:
        """ Determine the seed values, limits, and step sizes of parameters defined in the components.

        Note:
            These values should be specified with the proper names such that they will be recognized by Minuit.

        Args:
            user_arguments: User arguments to override the arguments to the fit.
        Returns:
            dict: Parameter values and limits in a dictionary suitable to be used as Minuit args.
        """
        arguments: FitArguments = {}
        for component in self.components.values():
            arguments.update(component.determine_parameters_limits())

        # Handle the user arguments
        # First, ensure that all user passed arguments are already in the argument keys. If not, the user probably
        # passed the wrong argument unintentionally.
        for k, v in user_arguments.items():
            if k not in arguments:
                raise ValueError(
                    f"User argument {k} (with value {v}) is not present in the fit arguments."
                    f" Possible arguments: {arguments}"
                )
        # Now, we actually assign the user arguments. We assign them last so we can overwrite any default arguments
        arguments.update(user_arguments)

        return arguments

    def _run_fit(self, arguments: FitArguments) -> Tuple[bool, iminuit.Minuit]:
        """ Make the proper calls to ``iminuit`` to run the fit.

        Args:
            arguments: Arguments for the ``minuit`` object based on seed values and limits specified in
                the fit components.
        Returns:
            tuple: (fitOkay, minuit) where fitOkay (bool) is ``True`` if the fit is okay, and
                ``minuit`` (``iminuit.minuit``) is the Minuit object which was used to perform the fit.
        """
        logger.debug(f"Minuit args: {arguments}")
        minuit = iminuit.Minuit(self._fit, **arguments)

        # Perform the fit
        minuit.migrad()
        # Run minos if requested.
        if self.use_minos:
            logger.info("Running MINOS. This may take a minute...")
            minuit.minos()
        # TEMP
        #minuit.print_matrix()
        #minuit.print_param()
        # ENDTEMP
        # Just in case (doesn't hurt anything, but may help in a few cases).
        minuit.hesse()
        # Plot the correlation matrix
        minuit.print_matrix()
        return (minuit.migrad_ok(), minuit)

    def fit(self, data: Union[InputData, Data], user_arguments: FitArguments = None) -> Tuple[bool, Data, iminuit.Minuit]:
        """ Perform the actual fit.

        Args:
            data: Input data to be used for the fit. The keys should either be of the form
                ``[region][orientation]`` or ``[FitType]``. The values can be uproot or ROOT 1D histograms.
            user_arguments: User arguments to override the arguments to the fit. Default: None.
        Returns:
            tuple: (fit_success, formatted_data, minuit) where fit_success (bool) is ``True`` if the fitting
                procedure was successful, formatted_data (dict) is the data reformatted in the preferred format for
                the fit, and minuit (iminuit.Minuit) is the minuit object, which is provided it is provided for
                specialized use cases. Note that the fit results (which contains most of the information in the minuit
                object) are stored in the class.
        """
        # Validate settings.
        if user_arguments is None:
            user_arguments = {}
        good_settings = self._validate_settings()
        if not good_settings:
            raise ValueError("Invalid settings! Please check the inputs.")

        # Setup and validate data.
        if not data:
            raise ValueError("Must pass data to be fitted!")
        formatted_data = self._format_input_data(data)
        good_data = self._validate_data(formatted_data)
        if not good_data:
            raise ValueError(
                f"Insufficient data provided for the fit components. Component keys: {self.components.keys()},"
                f" Data keys: {data.keys()}"
            )

        # Extract the x locations from where the fit should be evaluated.
        # Must be set before setting up the fit, which can limit the histogram x.
        data_temp = next(iter(formatted_data.values()))
        # Help out mypy
        assert isinstance(data_temp, histogram.Histogram1D)
        x = data_temp.x

        # Setup the fit components.
        fit_data: Data = {}
        for fit_type, component in self.components.items():
            fit_data[fit_type] = component._setup_fit(input_hist = formatted_data[fit_type])

        # Setup the final fit.
        self._fit = probfit.SimultaneousFit(*[component.cost_function for component in self.components.values()])
        arguments = self._determine_component_parameter_limits(user_arguments = user_arguments)

        # Perform the actual fit
        (good_fit, minuit) = self._run_fit(arguments = arguments)
        # Check if fit is considered valid
        if good_fit is False:
            raise base.FitFailed("Minimization failed! The fit is invalid!")
        # Check covariance matrix accuracy. We need to check it explictily because It appears that it is not
        # included in the migrad_ok status check.
        if not minuit.matrix_accurate:
            raise base.FitFailed("Corvairance matrix is not accurate!")

        # Calculate chi2/ndf (or really, min function value/ndf)
        # NDF = number of points used in the fit minus the number of free parameters.
        fixed_parameters = [k for k, v in minuit.fixed.items() if v is True]
        parameters = iminuit.util.describe(self._fit)
        # Can't just use set(parameters) - set(fixed_parameters) becase set() is unordered!
        free_parameters = [p for p in parameters if p not in set(fixed_parameters)]
        # Determine the number of fit data points. This cannot just be the length of x because we need to count
        # the data points that are used in all histograms!
        # For example, 36 data points / hist with the three orientation background fit should have
        # 18 (since near-side only) * 3 = 54 points.
        n_fit_data_points = sum(len(hist.x) for hist in fit_data.values())
        logger.debug(
            f"n_fit_data_points: {n_fit_data_points}, fixed_parameters: {fixed_parameters},"
            f" parameters: {parameters}, free_parameters: {free_parameters}"
        )

        # Store Minuit information for calculating the errors.
        self.fit_result = base.RPFitResult(
            parameters = parameters,
            fixed_parameters = fixed_parameters,
            free_parameters = free_parameters,
            values_at_minimum = dict(minuit.values),
            errors_on_parameters = dict(minuit.errors),
            covariance_matrix = minuit.covariance,
            x = x,
            n_fit_data_points = n_fit_data_points,
            minimum_val = minuit.fval,
        )
        for fit_type, component in self.components.items():
            self.components[fit_type].fit_result = base.ComponentFitResult.from_rp_fit_result(fit_result = self.fit_result,
                                                                                              component = component)
        logger.debug(f"nDOF: {self.fit_result.nDOF}")

        # Calculate the errors.
        for component in self.components.values():
            component.fit_result.errors = component.calculate_fit_errors(x = self.fit_result.x)

        # Return true to note success.
        return (True, formatted_data, minuit)

    def read_fit_results(self, filename: str, y: yaml.ruamel.yaml.YAML = None) -> bool:
        """ Read all fit results from the specified filename using YAML.

        We don't read the entire object from YAML because they we would have to deal with
        serializing ``probfit`` classes. Instead, we read the fit results, with the expectation
        that the fit object will be created independently, and then the results will be loaded.

        Args:
            filename: Name of the file under which the fit results are saved.
            y: YAML object to be used for reading the result. If none is specified, one will be created automatically.
        Returns:
            True if the reading was successful. The results will be read into the fit object.
        """
        # Create the YAML object if necessary.
        if y is None:
            y = yaml.yaml(modules_to_register = [base])

        with open(filename, "r") as f:
            fit_results = y.load(f)

        # Assign the full fit result.
        # We pop the value so that the can iterate over the remaining results.
        self.fit_result = fit_results.pop("full")
        # Assign to the components.
        for fit_type, result in fit_results.items():
            self.components[fit_type].fit_result = result

        return True

    def write_fit_results(self, filename: str, y: yaml.ruamel.yaml.YAML = None) -> bool:
        """ Write all fit results to the specified filename using YAML.

        We don't write the entire object to YAML because they we would have to deal with
        serializing ``probfit`` classes. Instead, we write the results, with the expectation
        that the fit object will be created independently, and then the results will be loaded.

        Args:
            filename: Name of the file under which the fit results will be saved.
            y: YAML object to be used for writing the result. If none is specified, one will be created automatically.
        Returns:
            True if the writing was successful.
        """
        # Create the YAML object if necessary.
        if y is None:
            y = yaml.yaml(modules_to_register = [base])

        # We need to create a dict to format the output because we don't want to have to deal
        # with registering `probfit` classes. "full" represent the full fit result, while the
        # fit component results are stored under their ``fit_type``. Note that we can't just
        # recreate the component fits from the full fit result because the full fit result doesn't
        # store the calculated errors. We could of course recalculate the errors, but that would
        # slow down loading the result, which would partially default the purpose. Plus, while
        # there is some duplicated information, it's really not that much storage space, even for
        # uncompressed information.
        output: Dict[Union[str, base.FitType], base.FitResult] = {"full": self.fit_result}
        output.update({fit_type: fit_component.fit_result for fit_type, fit_component in self.components.items()})
        with open(filename, "w+") as f:
            y.dump(output, f)

        return True

class FitComponent(ABC):
    """ A component of the fit.

    Args:
        fit_type (FitType): Type (region and orientation) of the fit component.
        resolution_parameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82"
        use_log_likelihood (bool): If true, use log likelihood cost function. Often used
            when statistics are limited. Default: False

    Attributes:
        fit_type (FitType): Overall type of the fit.
        use_log_likelihood (bool): True if the fit should be performed using log likelihood.
        fit_function (function): Function of the component.
        cost_function (probfit.costFunc): Cost function associated with the fit component.
    """
    def __init__(self, fit_type: base.FitType, resolution_parameters: ResolutionParameters, use_log_likelihood: bool = False) -> None:
        self.fit_type = fit_type
        self.use_log_likelihood = use_log_likelihood

        # Additional values
        # Will be determine in each subclass
        # Called last to ensure that all variables are available.
        self.fit_function: Callable[..., float]
        # Fit cost function
        self.cost_function = None
        # Background function. This describes the background of the component. In the case that the component
        # is fit to the background, this is identical to the fit function.
        self.background_function: Callable[..., float]

        # Result
        self.fit_result: base.ComponentFitResult

    def set_fit_type(self, region: Optional[str] = None, orientation: Optional[str] = None):
        """ Update the fit type.

        Will use the any of the existing parameters if they are not specified.

        Args:
            region (str): Region of the fit type.
            orientation (str): Orientation of the fit type.
        Returns:
            None: The fit type is set in the component.
        """
        if not region:
            region = self.fit_type.region
        if not orientation:
            orientation = self.fit_type.orientation
        self.fit_type = base.FitType(region = region, orientation = orientation)

    def evaluate_fit(self, x: np.ndarray) -> np.ndarray:
        """ Evaluate the fit component.

        Args:
            x: x values where the fit component will be evaluted.
        Returns:
            Function values at the given x values.
        """
        return probfit.nputil.vector_apply(self.fit_function, x, *list(self.fit_result.values_at_minimum.values()))

    def evaluate_background(self, x: np.ndarray) -> np.ndarray:
        """ Evaluates the background components of the fit function.

        In the case of a background component, this is identical to ``evaluate_fit(...)``. However, in the case of
        a signal component, this describes the background contribution to the signal.

        Args:
            x: x values where the fit component will be evaluted.
        Returns:
            Background function values at the given x values.
        """
        parameters = iminuit.util.describe(self.background_function)
        # NOTE: "x" is not in values_at_minimum (as expected), so we have to check that each parameter name
        #       is in values_at_minimum to prevent "x" from causing a problem
        return probfit.nputil.vector_apply(
            self.background_function,
            x, *[self.fit_result.values_at_minimum[p] for p in parameters if p in self.fit_result.values_at_minimum]
        )

    @property
    def rp_orientation(self) -> str:
        return self.fit_type.orientation

    @property
    def region(self) -> str:
        return self.fit_type.region

    @abstractmethod
    def determine_fit_function(self, resolution_parameters: ResolutionParameters, reaction_plane_parameter: base.ReactionPlaneParameter) -> None:
        """ Use the class parameters to determine the fit and background functions and store them. """
        ...

    def _setup_fit(self, input_hist: histogram.Histogram1D) -> histogram.Histogram1D:
        """ Setup the fit using information from the input hist.

        Args:
            input_hist (histogram.Histogram1D): The histogram to be fit by this function.
        Returns:
            histogram.Histogram1D: The data limited histogram (to be used for determining the number of data points used
                in the fit). Note that the fit is fully setup at this point.
        """
        # Determine the data which should be used for the fit.
        limited_hist = self.set_data_limits(hist = input_hist)

        # Determine the cost function
        self.cost_function = self._cost_function(hist = limited_hist)

        return limited_hist

    def set_data_limits(self, hist: histogram.Histogram1D) -> histogram.Histogram1D:
        """ Extract the data from the histogram. """
        return hist

    def _cost_function(self, hist: Optional[histogram.Histogram1D] = None, bin_edges: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, errors_squared: Optional[np.ndarray] = None):
        """ Define the cost function.

        Called when setting up a fit object.

        Note:
            We don't want to use the binned cost function versions - they will bin
            the data that is given, which is definitely not what we want. Instead,
            use the unbinned functions (as defined by ``probfit``).

        Note:
            Either specify the hist or the (bin_edges, y, errors_squared) tuple.

        Args:
            hist: Input histogram.
            bin_edges: The bin edges associated with an input histogram.
            y: The y values associated with an input histogram.
            errors_squared: The errors associated with an input histogram.
        Returns:
            probfit.costFunc: The defined cost function.
        """
        # Argument validation and properly format individual array is necessary.
        if not hist:
            # Validate the individual arguments
            if not bin_edges or not y or not errors_squared:
                raise ValueError("Must provide bin_edges, y, and errors_squared arrays.")
            # Then format them so they can be used.
            hist = histogram.Histogram1D(bin_edges = bin_edges, y = y, errors_squared = errors_squared)
        else:
            # These shouldn't be set if we're using a histogram.
            if bin_edges or y or errors_squared:
                raise ValueError(
                    "Provided histogram, and bin_edges, y, or errors_squared. Must provide only the histogram!"
                )

        if self.use_log_likelihood:
            logger.debug(f"Using log likelihood for {self.fit_type}, {self.rp_orientation}, {self.region}")
            # Generally will use when statistics are limited.
            # Errors are extracted by assuming a Poisson distribution, so we don't need to pass them explicitly (?)
            cost_function = probfit.UnbinnedLH(f = self.fit_function,
                                               data = hist.x)
        else:
            logger.debug(f"Using Chi2 for {self.fit_type}, {self.rp_orientation}, {self.region}")
            cost_function = probfit.Chi2Regression(f = self.fit_function,
                                                   x = hist.x,
                                                   y = hist.y,
                                                   error = hist.errors)

        # Return the function so that it can be stored. We explicitly return it
        # so that it is clear that it is being assigned.
        return cost_function

    def determine_parameters_limits(self) -> FitArguments:
        """ Determine the parameter seed values, limits, step sizes for the component.

        Note:
            These values should be specified with the proper names so that they will be recognized by Minuit.
            This is the users' responsbility.

        Args:
            None.
        Returns:
            Dictionary of the function arguments which specifies their
        """
        arguments: FitArguments = {}
        if self.region == "signal":
            # Signal default parameters
            # NOTE: The error should be approximately 10% of the value to ensure that the step size of the fit
            #       is correct.
            ns_amplitude = 10
            as_amplitude = 1
            ns_sigma_init = 0.15
            as_sigma_init = 0.3
            sigma_lower_limit = 0.02
            sigma_upper_limit = 0.7
            signal_limits: FitArguments = {
                "ns_amplitude": ns_amplitude, "limit_ns_amplitude": (0, 1000), "error_ns_amplitude": 0.1 * ns_amplitude,
                "as_amplitude": as_amplitude, "limit_as_amplitude": (0, 1000), "error_as_amplitude": 0.1 * as_amplitude,
                "ns_sigma": ns_sigma_init,
                "limit_ns_sigma": (sigma_lower_limit, sigma_upper_limit), "error_ns_sigma": 0.1 * ns_sigma_init,
                "as_sigma": as_sigma_init,
                "limit_as_sigma": (sigma_lower_limit, sigma_upper_limit), "error_as_sigma": 0.1 * as_sigma_init,
                "signal_pedestal": 0.0, "fix_signal_pedestal": True,
            }

            if self.rp_orientation != "inclusive":
                # Add the reaction plane prefix so the arguments don't conflict.
                signal_limits = iminuit.util.fitarg_rename(signal_limits, lambda name: self.rp_orientation + "_" + name)

            # Add in the signal limits regardless of RP orientation
            arguments.update(signal_limits)

            # Background label related to the fourier series (and this is labeled as "BG")
            background_label = "BG"
        else:
            # Background limits related to the RP background (and thus is labeled as "B")
            background_label = "B"

        # Now update the actual background limits
        signal_background_parameter_limits: FitArguments = {
            f"{background_label}": 10,
            f"limit_{background_label}": (0, 1000),
            f"error_{background_label}": 1
        }
        arguments.update(signal_background_parameter_limits)

        # Background function arguments
        # These should consistently be defined regardless of the fit definition.
        # Note that if the arguments already exist, they made be updated to the same value. However, this shouldn't
        # cause any problems of have any effect.
        # NOTE: The error should be approximately 10% of the value to ensure that the step size of the fit
        #       is correct.
        background_limits: FitArguments = {
            "v2_t": 0.02, "limit_v2_t": (0, 0.50), "error_v2_t": 0.001,
            "v2_a": 0.02, "limit_v2_a": (0, 0.50), "error_v2_a": 0.001,
            "v4_t": 0.01, "limit_v4_t": (0, 0.50), "error_v4_t": 0.001,
            "v4_a": 0.01, "limit_v4_a": (0, 0.50), "error_v4_a": 0.001,
            "v3": 0.0, "limit_v3": (-0.1, 0.5), "error_v3": 0.001,
            "v1": 0.0, "fix_v1": True,
        }
        arguments.update(background_limits)

        # Set error definition depending on whether we are using log likelihood or not
        # 0.5 should be used for negative log-likelihood, while 1 should be used for least sqaures (chi2)
        arguments.update({"errordef": 0.5 if self.use_log_likelihood else 1.0})

        return arguments

    def calculate_fit_errors(self, x: np.ndarray) -> np.ndarray:
        """ Calculate the fit function errors based on values from the fit.

        Args:
            x: x values where the errors will be evaluted.
        Returns:
            The calculated error values.
        """
        return self._calculate_function_errors(func = self.fit_function, x = x)

    def calculate_background_function_errors(self, x: np.ndarray) -> np.ndarray:
        """ Calculate the background function errors based on values from the fit.

        Args:
            x: x values where the errors will be evaluted.
        Returns:
            The calculated error values.
        """
        return self._calculate_function_errors(func = self.background_function, x = x)

    def _calculate_function_errors(self, func: Callable[..., float], x: np.ndarray) -> np.array:
        """ Calculate the errors of the given function based on values from the fit.

        Args:
            func: Function to use in calculating the errors. This should be a function stored in the component,
                such as the fit function or background function.
            x: x values where the errors will be evaluted.
        Returns:
            The calculated error values.
        """
        # Determine relevant parameters for the given function
        func_parameters = iminuit.util.describe(func)

        # Wrapper needed to call the function because ``numdifftools`` requires that multiple arguments
        # are in a single list. The wrapper expands that list for us.
        def func_wrap(x):
            # Need to expand the arguments
            return func(*x)
        # Setup to compute the derivative
        partial_derivative_func = nd.Gradient(func_wrap)

        # Determine the arguments for the fit function
        # Cannot use just values_at_minimum because x must be the first argument. So instead, we create the dict
        # with "x" as the first arg, and then update with the rest. We set it here to a very large float to be
        # clear that it will be set later.
        args_at_minimum = {"x": -1000000.0}
        args_at_minimum.update({k: v for k, v in self.fit_result.values_at_minimum.items() if k in func_parameters})
        logger.debug(f"args_at_minimum: {args_at_minimum}")
        # Retrieve the parameters to use in calculating the fit errors
        free_parameters = [p for p in self.fit_result.free_parameters if p in func_parameters]
        logger.debug(f"free_parameters: {free_parameters}")

        # To store the errors for each point
        error_vals = np.zeros(len(x))

        for i, val in enumerate(x):
            #logger.debug(f"val: {val}")
            # Add in x for the function call
            args_at_minimum["x"] = val

            # Evaluate the partial derivative at a point
            # We need to calculate the derivative once per x value
            # We time it to keep track of how long it takes to evaluate.
            start = time.time()
            logger.debug(f"Calculating the gradient for point {i}.")
            # The args must be called as a list.
            list_of_args_for_func_call = list(args_at_minimum.values())
            partial_derivative_result = partial_derivative_func(list_of_args_for_func_call)
            end = time.time()
            logger.debug(f"Finished calculating the graident in {end-start} seconds.")

            # Calculate error
            error_val = 0
            for i_name in free_parameters:
                for j_name in free_parameters:
                    # Determine the error value
                    i_name_index = list_of_args_for_func_call.index(args_at_minimum[i_name])
                    j_name_index = list_of_args_for_func_call.index(args_at_minimum[j_name])
                    #logger.debug(
                    #    f"Calculating error for i_name: {i_name}, i_name_index: {i_name_index}"
                    #    f" j_name: {j_name}, j_name_index: {j_name_index}"
                    #)

                    # Add error to overall error value
                    error_val += (
                        partial_derivative_result[i_name_index]
                        * partial_derivative_result[j_name_index]
                        * self.fit_result.covariance_matrix[(i_name, j_name)]
                    )

            # Modify from error squared to error
            error_val = np.sqrt(error_val)

            # Store
            #logger.debug("i: {}, error_val: {}".format(i, error_val))
            error_vals[i] = error_val

        return error_vals

class SignalFitComponent(FitComponent):
    """ Fit component in the signal region.

    Args:
        rp_orientation (str): Reaction plane orientation for the component.
        *args (list): Only use named args.
        **kwargs (dict): Named arguments to be passed on to the base component.
    """
    def __init__(self, rp_orientation, *args, **kwargs):
        # Validation
        if args:
            raise ValueError(f"Please specify all variables by name. Gave positional arguments: {args}")
        super().__init__(base.FitType(region = "signal", orientation = rp_orientation), **kwargs)

    def determine_fit_function(self, resolution_parameters: ResolutionParameters, reaction_plane_parameter: base.ReactionPlaneParameter) -> None:
        self.fit_function = functions.determine_signal_dominated_fit_function(
            rp_orientation = self.rp_orientation,
            resolution_parameters = resolution_parameters,
            reaction_plane_parameter = reaction_plane_parameter,
            rp_background_function = lambda: -1e6,  # Large negative number to ensure that it is clear that this went wrong
        )

class BackgroundFitComponent(FitComponent):
    """ Fit component in the background region.

    Args:
        rp_orientation (str): Reaction plane orientation for the component.
        *args (list): Only use named args.
        **kwargs (dict): Named arguments to be passed on to the base component.
    """
    def __init__(self, rp_orientation, *args, **kwargs):
        # Validation
        if args:
            raise ValueError(f"Please specify all variables by name. Gave positional arguments: {args}")
        super().__init__(base.FitType(region = "background", orientation = rp_orientation), **kwargs)

    def determine_fit_function(self, resolution_parameters: ResolutionParameters, reaction_plane_parameter: base.ReactionPlaneParameter) -> None:
        self.fit_function = functions.determine_background_fit_function(
            rp_orientation = self.rp_orientation,
            resolution_parameters = resolution_parameters,
            reaction_plane_parameter = reaction_plane_parameter,
            rp_background_function = lambda: -1e6,  # Large negative number to ensure that it is clear that this went wrong
        )

    def set_data_limits(self, hist: histogram.Histogram1D) -> histogram.Histogram1D:
        """ Set the limits of the fit to only use near-side data (ie dPhi < pi/2)

        Only the near-side will be used for the fit to avoid the signal at large
        dPhi on the away-side.

        Args:
            hist (histogram.Histogram1D): The input data.
        Returns:
            histogram.Histogram1D: The data limited to the near-side.
        """
        # Use only near-side data (ie dPhi < pi/2)
        ns_range = int(len(hist.x) / 2.0)
        # + 1 because we need to include the ending bin edge.
        bin_edges = hist.bin_edges[:ns_range + 1]
        y = hist.y[:ns_range]
        errors_squared = hist.errors_squared[:ns_range]

        # Return a data limits histogram
        return histogram.Histogram1D(bin_edges = bin_edges, y = y, errors_squared = errors_squared)
