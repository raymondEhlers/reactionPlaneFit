#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
from dataclasses import dataclass, field
import iminuit
import logging

logger = logging.getLogger(__name__)

class FitFailed(Exception):
    """ Raised if the fit failed. The message will include further details. """
    pass

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
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
    """
    parameters: list
    free_parameters: list
    fixed_parameters: list
    values_at_minimum: dict
    covariance_matrix: dict

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
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
        x (list): x values where the fit result should be evaluated.
        n_fit_data_points (int): Number of data points used in the fit.
        minimul_val (float): Minimum value of the fit when it coverages. This is the chi2 value for a
            chi2 minimization fit.
        components (dict): Contains fit results for the fit components. Most of the stored information is a subset
            of the information in this object, but it is much more convenient to have it accessible.
        nDOF (int): Number of degrees of freedom. Calculated on request from ``n_fit_data_points`` and ``free_parameters``.
    """
    x: list
    n_fit_data_points: int
    minimum_val: float
    components: dict = field(default_factory = dict)

    @property
    def nDOF(self):
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
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
        errors (dict): Store the errors associated with the component fit function. Keys are ``fit.FitType``,
            while values are arrays of the errors.
    """
    errors: dict = field(default_factory = dict)

    @classmethod
    def from_rp_fit_result(cls, fit_result: RPFitResult, component):
        """ Extract the component fit result from the fit component and the RP fit result.

        Args:
            fit_result (RPFitResult): Fit result from the RP fit.
            component (fit.FitComponent): Fit component for this fit result.
        Returns:
            ComponentFitResult: Constructed component fit result.
        """
        # We use the cost function because it describe will then exclude the x parameter (which is what we want)
        parameters = iminuit.util.describe(component.cost_function)
        # Pare down the values to only include parameters which are relevant for this component.
        fixed_parameters = [p for p in parameters if p in fit_result.fixed_parameters]
        free_parameters = [p for p in parameters if p not in fit_result.fixed_parameters]
        # Need to carefully grab the available values corresponding to the parameters or free_parameters, respectively.
        # NOTE: We cannot just iterate over the dicts themselves and check if the keys are in parameters because
        #       the parameters are deduplicated, and thus the order can be wrong. In particular, for signal fits,
        #       B of the background fit ends up at the end of the dict because all of the other parameters are already
        #       defined for the signal fit. This approach won't have a problem with this, because we retrieve the values
        #       in the order of the parameters of the current fit component.
        values_at_minimum = {p: fit_result.values_at_minimum[p] for p in parameters}
        covariance_matrix = {(a, b): fit_result.covariance_matrix[(a, b)] for a in free_parameters for b in free_parameters}

        return cls(
            parameters = parameters,
            free_parameters = free_parameters,
            fixed_parameters = fixed_parameters,
            values_at_minimum = values_at_minimum,
            covariance_matrix = covariance_matrix
        )

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

