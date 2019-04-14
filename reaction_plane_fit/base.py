#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
from dataclasses import dataclass
import iminuit
import logging
import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING

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

