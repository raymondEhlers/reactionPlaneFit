#!/usr/bine/env python

""" Defines the basic components of the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import iminuit
import numpy as np

import pachyderm.fit
from pachyderm import histogram
from pachyderm import yaml

from reaction_plane_fit import base
from reaction_plane_fit.base import InputData, Data
from reaction_plane_fit import functions

logger = logging.getLogger(__name__)

# Type helpers
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
        verbosity: Minuit verbosity level. Default: 3. This is extremely verbose, but it's quite useful.
                1 is commonly when you would like less verbose output.

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
        fit_result: Result of the RP fit.
        cost_func (Callable): Cost function for the RP fit.
    """
    # RP orientations (including inclusive). Should be overridden by the derived class.
    _rp_orientations: List[str] = []
    reaction_plane_parameters: Dict[str, base.ReactionPlaneParameter] = {}

    def __init__(self, resolution_parameters: ResolutionParameters,
                 use_log_likelihood: bool,
                 signal_region: Optional[Tuple[float, float]] = None,
                 background_region: Optional[Tuple[float, float]] = None,
                 use_minos: bool = False,
                 verbosity: int = 3):
        self.resolution_parameters = resolution_parameters
        self.use_log_likelihood = use_log_likelihood
        self.components: Dict[base.FitType, FitComponent] = {}
        self.regions = {"signal": signal_region, "background": background_region}
        self.use_minos = use_minos
        self.verbosity = verbosity

        # Contains the simultaneous fit to all of the components.
        self.cost_func = Callable[..., float]
        # Contains the fit results
        self.fit_result: base.FitResult

    @property
    def rp_orientations(self) -> List[str]:
        """ Get the RP orientations (excluding the inclusive, which is the last entry). """
        # "inclusive" must be the last entry.
        return self._rp_orientations[:-1]

    def _determine_reaction_plane_parameters(self, rp_orientation: str) -> base.ReactionPlaneParameter:
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
                reaction_plane_parameter = self._determine_reaction_plane_parameters(fit_type.orientation),
            )

        return True

    def _validate_settings(self) -> bool:
        """ Validate the passed settings. """
        # Check that there are sufficient resolution parameters.
        res_params = set(["R22", "R42", "R62", "R82"])
        good_params = res_params.issubset(self.resolution_parameters.keys())
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
        return base.format_input_data(data)

    def _validate_data(self, data: Data) -> bool:
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
        # Determine the initial set of arguments
        arguments: FitArguments = {}
        for component in self.components.values():
            arguments.update(component.determine_parameters_limits())
        # Set the Minuit verbosity. It can always be overridden by the user arguments
        arguments["print_level"] = self.verbosity

        # Handle the user arguments
        # First, ensure that all user passed arguments are already in the argument keys. If not, the user probably
        # passed the wrong argument unintentionally.
        for k, v in user_arguments.items():
            # The second condition allows us to fix components that were not previously fixed.
            if k not in arguments and k.replace("fix_", "") not in arguments:
                raise ValueError(
                    f"User argument {k} (with value {v}) is not present in the fit arguments."
                    f" Possible arguments: {arguments}"
                )
        # Now, we actually assign the user arguments. We assign them last so we can overwrite any default arguments
        arguments.update(user_arguments)

        return arguments

    def _setup_fit(self, data: Union[InputData, Data],
                   user_arguments: Optional[FitArguments] = None) -> Tuple[np.ndarray, Data, Data,
                                                                           FitArguments, pachyderm.fit.SimultaneousFit]:
        """ Complete all setup steps up to actually fitting the data.

        This includes formatting the input data, setting up the cost function, and determining
        the proper user arguments.

        Args:
            data: The input data.
            user_arguments: The user arguments that will be passed to the fit.
        Returns:
            (x, formatted_data, fit_data, user_arguments, cost_func) where x is the x values where the fit
                should be evaluated, formatted_data is the input data formatted into the proper format
                for use with the fit, fit_data is the formatted_data restricted to the fit ranges,
                user_arguments are the full set of user arguments for the fit, and cost_func is the
                simultaneous fit cost function to be stored in the fit object.
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
                f"Insufficient data provided for the fit components.\nComponent keys: {self.components.keys()}\n"
                f"Data keys: {data.keys()}\n"
                f"Formatted Data keys: {formatted_data.keys()}"
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

        # Setup the final fit to be performed simultaneously.
        #self.cost_func = sum(reversed(list(component.cost_function for component in self.components.values())))
        cost_func = sum(component.cost_function for component in self.components.values())
        # Help out mypy...
        assert isinstance(cost_func, pachyderm.fit.SimultaneousFit)
        arguments = self._determine_component_parameter_limits(user_arguments = user_arguments)

        return x, formatted_data, fit_data, arguments, cost_func

    def _run_fit(self, arguments: FitArguments, skip_hesse: bool) -> Tuple[bool, iminuit.Minuit]:
        """ Make the proper calls to ``iminuit`` to run the fit.

        Args:
            arguments: Arguments for the ``minuit`` object based on seed values and limits specified in
                the fit components.
            skip_hesse: This should be used rarely and only with care! True if the hesse should _not_ be explicitly run.
                This is useful because sometimes it fails without a clear reason. If skipped, Minos will automatically
                be run, and it's up to the user to ensure that the symmetric error estimates from Hesse run during the
                minimization are close to the Minos errors. If they're not, it indicates that something went wrong in
                the error calculation and Hesse was failing for good reason. Default: False.
        Returns:
            tuple: (fitOkay, minuit) where fitOkay (bool) is ``True`` if the fit is okay, and
                ``minuit`` (``iminuit.minuit``) is the Minuit object which was used to perform the fit.
        """
        # Setup the fit
        logger.debug(f"Minuit args: {arguments}")
        minuit = iminuit.Minuit(self.cost_func, **arguments)
        # Improve minimization reliability
        minuit.set_strategy(2)

        # Perform the fit
        # NOTE: This will plot the parameters deteremined by the fit.
        minuit.migrad()
        # Run minos if requested (or if HESSE is skipped so a cross check is available).
        if self.use_minos or skip_hesse:
            logger.info("Running MINOS. This may take a minute...")
            minuit.minos()
        # Just in case (usually doesn't hurt anything, but may help in a few cases).
        if not skip_hesse:
            # NOTE: This will print the HESSE calculated errors and the correlation matrix.
            minuit.hesse()
        else:
            # Since HESSE won't run and consequently print the final result, we call it be hand.
            minuit.print_param()

        return (minuit.migrad_ok(), minuit)

    def fit(self, data: Union[InputData, Data], user_arguments: Optional[FitArguments] = None,
            skip_hesse: bool = False) -> Tuple[bool, Data, iminuit.Minuit]:
        """ Perform the actual fit.

        Args:
            data: Input data to be used for the fit. The keys should either be of the form
                ``[region][orientation]`` or ``[FitType]``. The values can be uproot or ROOT 1D histograms.
            user_arguments: User arguments to override the arguments to the fit. Default: None.
            skip_hesse: This should be used rarely and only with care! True if the hesse should _not_ be explicitly run.
                This is useful because sometimes it fails without a clear reason. If skipped, Minos will automatically
                be run, and it's up to the user to ensure that the symmetric error estimates from Hesse run during the
                minimization are close to the Minos errors. If they're not, it indicates that something went wrong in
                the error calculation and Hesse was failing for good reason. Default: False.
        Returns:
            tuple: (fit_success, formatted_data, minuit) where fit_success (bool) is ``True`` if the fitting
                procedure was successful, formatted_data (dict) is the data reformatted in the preferred format for
                the fit, and minuit (iminuit.Minuit) is the minuit object, which is provided it is provided for
                specialized use cases. Note that the fit results (which contains most of the information in the minuit
                object) are stored in the class.
        """
        # Validte and setup the fit, storing the simultaneous fit cost function
        x, formatted_data, fit_data, arguments, self.cost_func = self._setup_fit(data = data, user_arguments = user_arguments)

        # Perform the actual fit
        (good_fit, minuit) = self._run_fit(arguments = arguments, skip_hesse = skip_hesse)
        # Check if fit is considered valid
        if good_fit is False:
            raise base.FitFailed("Minimization failed! The fit is invalid!")
        # Check covariance matrix accuracy. We need to check it explicitly because It appears that it is not
        # included in the migrad_ok status check.
        if not minuit.matrix_accurate():
            raise base.FitFailed("Corvairance matrix is not accurate!")

        # Determine some of the fit result parameters.
        fixed_parameters = [k for k, v in minuit.fixed.items() if v is True]
        parameters = iminuit.util.describe(self.cost_func)
        # Can't just use set(parameters) - set(fixed_parameters) because set() is unordered!
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
        self.fit_result = base.FitResult(
            parameters = parameters, fixed_parameters = fixed_parameters, free_parameters = free_parameters,
            values_at_minimum = dict(minuit.values), errors_on_parameters = dict(minuit.errors),
            covariance_matrix = minuit.covariance,
            x = x,
            n_fit_data_points = n_fit_data_points, minimum_val = minuit.fval,
            # This doesn't need to be meaningful - it won't be accessed.
            errors = [],
        )
        for fit_type, component in self.components.items():
            self.components[fit_type].fit_result = base.component_fit_result_from_rp_fit_result(
                fit_result = self.fit_result, component = component,
            )
        logger.debug(f"nDOF: {self.fit_result.nDOF}")

        # Calculate the errors.
        for component in self.components.values():
            component.fit_result.errors = component.calculate_fit_errors(x = self.fit_result.x)

        # Return true to note success.
        return (True, formatted_data, minuit)

    def read_fit_object(self, filename: str, data: Union[InputData, Data],
                        user_arguments: Optional[FitArguments] = None,
                        y: yaml.ruamel.yaml.YAML = None) -> Tuple[bool, Data]:
        """ Read the fit results and setup the entire RPF object.

        Using this method, the RPF object should be fully setup. The only step that isn't performed
        is actually running the fit using Minuit. To read just the fit results into the object,
        use ``read_fit_results(...)``.

        Args:
            filename: Name of the file under which the fit results are saved.
            data: Input data that was used for the fit. The keys should either be of the form
                ``[region][orientation]`` or ``[FitType]``. The values can be uproot or ROOT 1D histograms.
            user_arguments: User arguments to override the arguments to the fit. Default: None.
            y: YAML object to be used for reading the result. If none is specified, one will be created automatically.
        Returns:
            (read_success, formatted_data) where read_success is True if the reading was successful, and formatted_data
                is the input data reformatted according to the fit object format (ie. same as formatted_data
                returned when performing the fit). The results will be read into the fit object.
        """
        read_results = self.read_fit_results(filename = filename, y = y)
        if not read_results:
            raise RuntimeError(f"Unable to read fit results from {filename}. Check the input file.")

        # Fully setup the fit object.
        x, formatted_data, fit_data, arguments, self.cost_func = self._setup_fit(data = data, user_arguments = user_arguments)

        return True, formatted_data

    def read_fit_results(self, filename: str, y: yaml.ruamel.yaml.YAML = None) -> bool:
        """ Read all fit results from the specified filename using YAML.

        We don't read the entire object from YAML because they we would have to deal with
        serializing many classes. Instead, we read the fit results, with the expectation
        that the fit object will be created independently, and then the results will be loaded.

        To reconstruct the entire fit object, use ``read_fit_object(...)``.

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
        serializing many classes. Instead, we write the results, with the expectation
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
        # with registering many classes. "full" represent the full fit result, while the
        # fit component results are stored under their ``fit_type``. Note that we can't just
        # recreate the component fits from the full fit result because the full fit result doesn't
        # store the calculated errors. We could of course recalculate the errors, but that would
        # slow down loading the result, which would partially default the purpose. Plus, while
        # there is some duplicated information, it's really not that much storage space, even for
        # uncompressed information.
        output: Dict[Union[str, base.FitType], Union[base.BaseFitResult, base.FitResult]] = {"full": self.fit_result}
        output.update({fit_type: fit_component.fit_result for fit_type, fit_component in self.components.items()})
        logger.debug(f"output: {output}")

        # Create the directory if it doesn't exist, and then write the file.
        Path(filename).parent.mkdir(parents = True, exist_ok = True)
        with open(filename, "w+") as f:
            y.dump(output, f)

        return True

    @abstractmethod
    def create_full_set_of_components(self, input_data: Data) -> Dict[str, "FitComponent"]:
        """ Create the full set of fit components. """
        ...

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
        cost_func (pachyderm.fit.CostFunctionBase): Cost function associated with the fit component.
    """
    def __init__(self, fit_type: base.FitType, resolution_parameters: ResolutionParameters, use_log_likelihood: bool = False) -> None:
        self.fit_type = fit_type
        self.use_log_likelihood = use_log_likelihood

        # Additional values
        # Will be determine in each subclass
        # Called last to ensure that all variables are available.
        self.fit_function: Callable[..., float]
        # Fit cost function
        self.cost_function: pachyderm.fit.CostFunctionBase
        # Background function. This describes the background of the component. In the case that the component
        # is fit to the background, this is identical to the fit function.
        self.background_function: Callable[..., float]

        # Result
        self.fit_result: base.BaseFitResult

    def set_fit_type(self, region: Optional[str] = None, orientation: Optional[str] = None) -> None:
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
            x: x values where the fit component will be evaluated.
        Returns:
            Function values at the given x values.
        """
        return self.fit_function(x, *list(self.fit_result.values_at_minimum.values()))

    def evaluate_background(self, x: np.ndarray) -> np.ndarray:
        """ Evaluates the background components of the fit function.

        In the case of a background component, this is identical to ``evaluate_fit(...)``. However, in the case of
        a signal component, this describes the background contribution to the signal.

        Args:
            x: x values where the fit component will be evaluated.
        Returns:
            Background function values at the given x values.
        """
        parameters = iminuit.util.describe(self.background_function)
        # NOTE: We only need a subset of parameters to evaluate the background function, so we explicitly select them.
        return self.background_function(
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
            input_hist: The histogram to be fit by this function.
        Returns:
            The data limited histogram (to be used for determining the number of data points used in the fit).
                Note that the fit is fully setup at this point.
        """
        # Determine the data which should be used for the fit.
        limited_hist = self.set_data_limits(hist = input_hist)

        # Determine the cost function
        self.cost_function = self._cost_function(hist = limited_hist)

        return limited_hist

    def set_data_limits(self, hist: histogram.Histogram1D) -> histogram.Histogram1D:
        """ Extract the data from the histogram. """
        return hist

    def _cost_function(self, hist: Optional[histogram.Histogram1D] = None,
                       bin_edges: Optional[np.ndarray] = None,
                       y: Optional[np.ndarray] = None,
                       errors_squared: Optional[np.ndarray] = None) -> pachyderm.fit.CostFunctionBase:
        """ Define the cost function.

        Called when setting up a fit object.

        Note:
            Either specify the hist or the (bin_edges, y, errors_squared) tuple.

        Args:
            hist: Input histogram.
            bin_edges: The bin edges associated with an input histogram.
            y: The y values associated with an input histogram.
            errors_squared: The errors associated with an input histogram.
        Returns:
            The defined cost function.
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

        cost_func_class: Type[pachyderm.fit.CostFunctionBase]
        if self.use_log_likelihood:
            logger.debug(f"Using log likelihood for {self.fit_type}, {self.rp_orientation}, {self.region}")
            # Generally will use when statistics are limited.
            cost_func_class = pachyderm.fit.BinnedLogLikelihood
        else:
            logger.debug(f"Using Chi2 for {self.fit_type}, {self.rp_orientation}, {self.region}")
            cost_func_class = pachyderm.fit.BinnedChiSquared
        cost_func: pachyderm.fit.CostFunctionBase = cost_func_class(
            f = self.fit_function, data = hist
        )

        # Return the cost function class so that it can be stored. We explicitly return it
        # so that it is clear that it is being assigned.
        return cost_func

    def determine_parameters_limits(self) -> FitArguments:
        """ Determine the parameter seed values, limits, step sizes for the component.

        Note:
            These values should be specified with the proper names so that they will be recognized by Minuit.
            This is the user's responsibility.

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
            ns_amplitude = 1000
            as_amplitude = 1000
            ns_sigma_init = 0.15
            as_sigma_init = 0.3
            sigma_lower_limit = 0.02
            sigma_upper_limit = 0.7
            signal_limits: FitArguments = {
                "ns_amplitude": ns_amplitude, "limit_ns_amplitude": (0, 1e7), "error_ns_amplitude": 0.1 * ns_amplitude,
                "as_amplitude": as_amplitude, "limit_as_amplitude": (0, 1e7), "error_as_amplitude": 0.1 * as_amplitude,
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

            # Specify the background level associated with the signal function.
            # If using a separate fourier series, this should be labeled as "BG". We will
            # know this if the function includes "BG". Otherewise, just use the overall
            # backgrond level.
            background_label = "B"
            func_args = iminuit.util.describe(self.fit_function)
            if "BG" in func_args:
                background_label = "BG"
        else:
            # Background limits related to the RP background (and thus is labeled as "B")
            background_label = "B"

        # Now update the actual background limits
        signal_background_parameter_limits: FitArguments = {
            f"{background_label}": 10,
            f"limit_{background_label}": (0, 1e6),
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
            "v2_t": 0.02, "limit_v2_t": (0.001, 0.20), "error_v2_t": 0.001,
            "v2_a": 0.10, "limit_v2_a": (0.03, 0.50), "error_v2_a": 0.001,
            "v4_t": 0.005, "limit_v4_t": (0, 0.50), "error_v4_t": 0.0005,
            "v4_a": 0.01, "limit_v4_a": (0, 0.50), "error_v4_a": 0.001,
            # v3 is expected to be be 0. v3_t may be negative, so it's difficult to predict.
            "v3": 0.001, "limit_v3": (-1.0, 1.0), "error_v3": 0.0001,
            "v1": 0.0, "limit_v1": (-1.0, 1.0), "error_v1": 1e-4, "fix_v1": True,
        }
        arguments.update(background_limits)

        # Set error definition depending on whether we are using log likelihood or not
        # 0.5 should be used for negative log-likelihood, while 1 should be used for least squares (chi2)
        arguments.update({"errordef": 0.5 if self.use_log_likelihood else 1.0})

        return arguments

    def calculate_fit_errors(self, x: np.ndarray) -> np.ndarray:
        """ Calculate the fit function errors based on values from the fit.

        Args:
            x: x values where the errors will be evaluated.
        Returns:
            The calculated error values.
        """
        return base.calculate_function_errors(func = self.fit_function, fit_result = self.fit_result, x = x)

    def calculate_background_function_errors(self, x: np.ndarray) -> np.ndarray:
        """ Calculate the background function errors based on values from the fit.

        Args:
            x: x values where the errors will be evaluated.
        Returns:
            The calculated error values.
        """
        return base.calculate_function_errors(func = self.background_function, fit_result = self.fit_result, x = x)

class SignalFitComponent(FitComponent):
    """ Fit component in the signal region.

    Args:
        rp_orientation (str): Reaction plane orientation for the component.
        *args (list): Only use named args.
        **kwargs (dict): Named arguments to be passed on to the base component.
    """
    def __init__(self, rp_orientation: str, *args: Any, **kwargs: Any):
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
            inclusive_background_function = lambda: -1e6,  # Large negative number to ensure that it is clear that this went wrong
        )
        self.background_function = lambda: -1e6  # Large negative number to ensure that it is clear that this went wrong

class BackgroundFitComponent(FitComponent):
    """ Fit component in the background region.

    Args:
        rp_orientation (str): Reaction plane orientation for the component.
        *args (list): Only use named args.
        **kwargs (dict): Named arguments to be passed on to the base component.
    """
    def __init__(self, rp_orientation: str, *args: Any, **kwargs: Any):
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
            inclusive_background_function = lambda: -1e6,  # Large negative number to ensure that it is clear that this went wrong
        )
        self.background_function = lambda: -1e6  # Large negative number to ensure that it is clear that this went wrong

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
        ## This can be used to select a range between a specific set of values.
        #min_x, max_x = -np.pi / 2, np.pi / 2
        #selected_range = slice(hist.find_bin(min_x + epsilon), hist.find_bin(max_x - epsilon) + 1)
        #bin_edges_selected_range = ((hist.bin_edges >= min_x) & (hist.bin_edges <= max_x))
        ## Drop the furthest out bin
        #bin_edges = hist.bin_edges[bin_edges_selected_range]
        #y = hist.y[selected_range]
        #errors_squared = hist.errors_squared[selected_range]

        # Return a data limits histogram
        return histogram.Histogram1D(bin_edges = bin_edges, y = y, errors_squared = errors_squared)
