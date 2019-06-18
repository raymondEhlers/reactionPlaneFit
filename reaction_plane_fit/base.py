#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import iminuit
import logging
import numpy as np
from typing import TYPE_CHECKING

import pachyderm.fit as fit_base
from pachyderm import histogram
from pachyderm.fit.base import calculate_function_errors, FitFailed, FitResult  # noqa: F401

if TYPE_CHECKING:
    from reaction_plane_fit import fit

logger = logging.getLogger(__name__)

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

@dataclass(frozen = True)
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

def component_fit_result_from_rp_fit_result(fit_result: fit_base.FitResult,
                                            component: "fit.FitComponent",
                                            fit_data: histogram.Histogram1D) -> fit_base.FitResult:
    """ Create a component fit result from the fit component and the RP fit result.

    Note:
        The component fit result only contains the subset of fit information that is relevant for this component,
        even if it is part of a larger fit. This means that one must be careful when using the ``x``,
        ``n_fit_data_points``, and ``minimum_val`` of a component which is part of the larger fit. Better is to use
        the quantities calculated for the main fit.

    Args:
        fit_result: Fit result from the RP fit.
        component: Fit component for this fit result.
        fit_data: Fit data used for the fit component.
    Returns:
        Constructed component fit result.
    """
    # Although the cost function could be convenient (because it doesn't include the x parameter that we want
    # to exclude), the fit function is more appropriate.
    parameters = iminuit.util.describe(component.fit_function)
    parameters.pop(parameters.index("x"))
    # Pare down the values to only include parameters which are relevant for this component.
    #constructing_fake_component = False
    #storage_parameters = {p: p for p in parameters}
    #if "BG" in parameters and "BG" not in fit_result.values_at_minimum:
    #    logger.info("Faking component")
    #    constructing_fake_component = True
    #    #parameters[parameters.index("BG")] = "B"
    #    new_parameters = {}
    #    for p in parameters:
    #        if p == "BG":
    #            new_parameters["BG"] = "B"
    #        else:
    #            new_parameters[p] = p
    #    storage_parameters = new_parameters

    #logger.info(f"parameters: {parameters}")
    #logger.info(f"storage_parameters: {storage_parameters}")

    fixed_parameters = [p for p in parameters if p in fit_result.fixed_parameters]
    free_parameters = [p for p in parameters if p not in fit_result.fixed_parameters]
    # Need to carefully grab the available values corresponding to the parameters or free_parameters, respectively.
    # NOTE: We cannot just iterate over the dict(s) themselves and check if the keys are in parameters because
    #       the parameters are de-duplicated, and thus the order can be wrong. In particular, for signal fits,
    #       B of the background fit ends up at the end of the dict because all of the other parameters are already
    #       defined for the signal fit. This approach won't have a problem with this, because we retrieve the values
    #       in the order of the parameters of the current fit component.
    #values_at_minimum = {p: fit_result.values_at_minimum[storage_parameters[p]] for p in parameters}
    #errors_on_parameters = {p: fit_result.errors_on_parameters[storage_parameters[p]] for p in parameters}
    values_at_minimum = {p: fit_result.values_at_minimum[p] for p in parameters}
    errors_on_parameters = {p: fit_result.errors_on_parameters[p] for p in parameters}
    covariance_matrix = {
        (a, b): fit_result.covariance_matrix[(a, b)] for a in free_parameters for b in free_parameters
    }
    #logger.info(f"values_at_minimum: {values_at_minimum}")
    #logger.info(f"errors_on_parameters: {errors_on_parameters}")
    #logger.info(f"covariance_matrix: {covariance_matrix}")

    cost_function_data = component.cost_function.data
    # Help out mypy...
    assert isinstance(cost_function_data, histogram.Histogram1D)
    return fit_base.FitResult(
        parameters = parameters,
        free_parameters = free_parameters,
        fixed_parameters = fixed_parameters,
        values_at_minimum = values_at_minimum,
        errors_on_parameters = errors_on_parameters,
        covariance_matrix = covariance_matrix,
        # These are only the information relevant to this component, but it's derived from the main fit information,
        # so some care is required.
        x = cost_function_data.x,
        n_fit_data_points = len(cost_function_data.x),
        minimum_val = float(component.cost_function(*list(values_at_minimum.values()))),
        # This will be determined and set later.
        errors = np.array([]),
    )

