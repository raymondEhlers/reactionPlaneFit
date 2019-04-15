#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
from dataclasses import dataclass
import iminuit
import logging
import numdifftools as nd
import numpy as np
import time
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from reaction_plane_fit import fit

logger = logging.getLogger(__name__)

class FitFailed(Exception):
    """ Raised if the fit failed. The message will include further details. """
    pass

@dataclass(frozen = True)
class FitType:
    """ Describes the fit parameters of a particular component.

    Attributes:
        region: Describes the region in which the data for the fit originates. It should be either "signal" or
            "background" dominated.
        orientation: Describe the reaction plane orientation of the data. For data which does not select or orientation,
            it should be described as "inclusive". Otherwise, the values are up to the particular implementation. As an
            example, for three RP orientations, they are known as "in_plane", "mid_plane", and "out_of_plane".
    """
    region: str
    orientation: str

@dataclass
class FitResult(ABC):
    """ Fit result base class.

    Defined as an ABC because the derived class should be used (this class doesn't contain enough information on it's
    own to be useful as a fit result).

    Attributes:
        parameters (list): Names of the parameters used in the fit.
        free_parameters (list): Names of the free parameters used in the fit.
        fixed_parameters (list): Names of the fixed parameters used in the fit.
        values_at_minimum (dict): Contains the values of the full RP fit function at the minimum. Keys are the
            names of parameters, while values are the numerical values at convergence.
        errors_on_parameters (dict): Contains the values of the errors associated with the parameters
            determined via the fit.
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
    """
    parameters: List[str]
    free_parameters: List[str]
    fixed_parameters: List[str]
    values_at_minimum: Dict[str, float]
    errors_on_parameters: Dict[str, float]
    covariance_matrix: Dict[Tuple[str, str], float]

    @property
    def correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """ The correlation matrix of the free parameters.

        These values are derived from the covariance matrix values stored in the fit.

        Note:
            This property caches the correlation matrix value so we don't have to calculate it every time.

        Args:
            None
        Returns:
            The correlation matrix of the fit result.
        """
        try:
            return self._correlation_matrix
        except AttributeError:
            def corr(i_name: str, j_name: str) -> float:
                """ Calculate the correlation matrix (definition from iminuit) from the covariance matrix. """
                # The + 1e-100 is just to ensure that we don't divide by 0.
                value = (self.covariance_matrix[(i_name, j_name)]
                         / (np.sqrt(self.covariance_matrix[(i_name, i_name)]
                            * self.covariance_matrix[(j_name, j_name)]) + 1e-100)
                         )
                # Need to explicitly cast to float. Otherwise, it will return a np.float64, which will cause problems
                # for YAML...
                return float(value)

            matrix: Dict[Tuple[str, str], float] = {}
            for i_name in self.free_parameters:
                for j_name in self.free_parameters:
                    matrix[(i_name, j_name)] = corr(i_name, j_name)

            self._correlation_matrix = matrix

        return self._correlation_matrix

@dataclass
class RPFitResult(FitResult):
    """ Store the main RP fit result and the component results.

    Note:
        free_parameters + fixed_parameters = parameters

    Attributes:
        parameters (list): Names of the parameters used in the fit.
        free_parameters (list): Names of the free parameters used in the fit.
        fixed_parameters (list): Names of the fixed parameters used in the fit.
        values_at_minimum (dict): Contains the values of the full RP fit function at the minimum. Keys are the
            names of parameters, while values are the numerical values at convergence.
        errors_on_parameters (dict): Contains the values of the errors associated with the parameters
            determined via the fit.
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
        x: x values where the fit result should be evaluated.
        n_fit_data_points (int): Number of data points used in the fit.
        minimul_val (float): Minimum value of the fit when it coverages. This is the chi squared value for a
            chi squared minimization fit.
        nDOF (int): Number of degrees of freedom. Calculated on request from ``n_fit_data_points`` and ``free_parameters``.
    """
    x: np.array
    n_fit_data_points: int
    minimum_val: float

    @property
    def nDOF(self) -> int:
        return self.n_fit_data_points - len(self.free_parameters)

@dataclass
class ComponentFitResult(FitResult):
    """ Store fit component fit results.

    It is best to construct these fit results from the main fit results and the fit corresponding fit component
    via ``from_rp_fit_result(...)``.

    Attributes:
        parameters (list): Names of the parameters used in the fit.
        free_parameters (list): Names of the free parameters used in the fit.
        fixed_parameters (list): Names of the fixed parameters used in the fit.
        values_at_minimum (dict): Contains the values of the full RP fit function at the minimum. Keys are the
            names of parameters, while values are the numerical values at convergence.
        errors_on_parameters (dict): Contains the values of the errors associated with the parameters
            determined via the fit.
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
        errors: Store the errors associated with the component fit function.
    """
    errors: np.ndarray

    @classmethod
    def from_rp_fit_result(cls, fit_result: RPFitResult, component: "fit.FitComponent"):
        """ Extract the component fit result from the fit component and the RP fit result.

        Args:
            fit_result: Fit result from the RP fit.
            component: Fit component for this fit result.
        Returns:
            ComponentFitResult: Constructed component fit result.
        """
        # We use the cost function because it describe will then exclude the x parameter (which is what we want)
        parameters = iminuit.util.describe(component.cost_function)
        # Pare down the values to only include parameters which are relevant for this component.
        fixed_parameters = [p for p in parameters if p in fit_result.fixed_parameters]
        free_parameters = [p for p in parameters if p not in fit_result.fixed_parameters]
        # Need to carefully grab the available values corresponding to the parameters or free_parameters, respectively.
        # NOTE: We cannot just iterate over the dict(s) themselves and check if the keys are in parameters because
        #       the parameters are de-duplicated, and thus the order can be wrong. In particular, for signal fits,
        #       B of the background fit ends up at the end of the dict because all of the other parameters are already
        #       defined for the signal fit. This approach won't have a problem with this, because we retrieve the values
        #       in the order of the parameters of the current fit component.
        values_at_minimum = {p: fit_result.values_at_minimum[p] for p in parameters}
        errors_on_parameters = {p: fit_result.errors_on_parameters[p] for p in parameters}
        covariance_matrix = {
            (a, b): fit_result.covariance_matrix[(a, b)] for a in free_parameters for b in free_parameters
        }

        return cls(
            parameters = parameters,
            free_parameters = free_parameters,
            fixed_parameters = fixed_parameters,
            values_at_minimum = values_at_minimum,
            errors_on_parameters = errors_on_parameters,
            covariance_matrix = covariance_matrix,
            # This will be determine and set later.
            errors = np.array([]),
        )

def calculate_function_errors(func: Callable[..., float], fit_result: FitResult, x: np.ndarray) -> np.array:
    """ Calculate the errors of the given function based on values from the fit.

    Note:
        We don't take the x values for the fit_result as it may be desirable to calculate the errors for
        only a subset of x values. Plus, the component fit result doesn't store the x values, so it would
        complicate the validation. It's much easier to just require the user to pass the x values (and it takes
        little effort to do so).

    Args:
        func: Function to use in calculating the errors.
        fit_result: Fit result for which the errors will be calculated.
        x: x values where the errors will be evaluated.
    Returns:
        The calculated error values.
    """
    # Determine relevant parameters for the given function
    func_parameters = iminuit.util.describe(func)

    # We need a function wrapper to call our fit function because ``numdifftools`` requires that each variable
    # which will be differentiated against must in a list in the first argument. The wrapper just expands
    # that list for us.
    def func_wrap(x):
        # Need to expand the arguments
        return func(*x)
    # Setup to compute the derivative
    partial_derivative_func = nd.Gradient(func_wrap)

    # Determine the arguments for the fit function
    # NOTE: The fit result may have more arguments at minimum and free parameters than the fit function that we've
    #       passed (for example, if we've calculating the background parameters for the inclusive signal fit), so
    #       we need to determine the free parameters here.
    # We cannot use just values_at_minimum because the arguments are ordered and therefore x must be the first argument.
    # So instead, we create the dict with "x" as the first key, and then update with the rest. We set it here to a very
    # large float to be clear that it will be set later.
    args_at_minimum = {"x": -1000000.0}
    args_at_minimum.update({k: v for k, v in fit_result.values_at_minimum.items() if k in func_parameters})
    # Retrieve the parameters to use in calculating the fit errors.
    free_parameters = [p for p in fit_result.free_parameters if p in func_parameters]
    # To calculate the error, we need to match up the parameter names to their index in the arguments list
    args_at_minimum_keys = list(args_at_minimum)
    name_to_index = {name: args_at_minimum_keys.index(name) for name in free_parameters}
    logger.debug(f"args_at_minimum: {args_at_minimum}")
    logger.debug(f"free_parameters: {free_parameters}")
    logger.debug(f"name_to_index: {name_to_index}")

    # To store the errors for each point
    error_vals = np.zeros(len(x))

    for i, val in enumerate(x):
        # Specify x for the function call
        args_at_minimum["x"] = val

        # Evaluate the partial derivative at a given x value with respect to all of the variables in the given function.
        # In principle, we've doing some unnecessary work because we also calculate the gradient with
        # respect to fixed parameters. But due to the argument requirements of ``numdifftools``, it would be
        # quite difficult to tell it to only take the gradient with respect to a non-continuous selection of
        # parameters. So we just accept the inefficiency.
        logger.debug(f"Calculating the gradient for point {i}.")
        # We time it to keep track of how long it takes to evaluate. Sometimes it can be a bit slow.
        start = time.time()
        # Actually evaluate the gradient. The args must be called as a list.
        partial_derivative_result = partial_derivative_func(list(args_at_minimum.values()))
        end = time.time()
        logger.debug(f"Finished calculating the gradient in {end-start} seconds.")

        # Finally, calculate the error by multiplying the matrix of gradient by the covariance matrix values.
        error_val = 0
        for i_name in free_parameters:
            for j_name in free_parameters:
                # Determine the error value
                #logger.debug(f"Calculating error for i_name: {i_name}, j_name: {j_name}")
                # Add error to overall error value
                error_val += (
                    partial_derivative_result[name_to_index[i_name]]
                    * partial_derivative_result[name_to_index[j_name]]
                    * fit_result.covariance_matrix[(i_name, j_name)]
                )

        # Store the value at the specified point. Note that we store the error, rather than the error squared.
        # Modify from error squared to error
        #logger.debug("i: {}, error_val: {}".format(i, error_val))
        error_vals[i] = np.sqrt(error_val)

    return error_vals

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

