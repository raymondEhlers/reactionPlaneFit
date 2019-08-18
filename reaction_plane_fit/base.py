#!/usr/bin/env python

""" Utility functions for the reaction plane fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import iminuit
import logging
import numpy as np
from typing import Dict, Union, TYPE_CHECKING, cast

import pachyderm.fit as fit_base
from pachyderm.fit.base import calculate_function_errors, FitFailed, BaseFitResult, FitResult  # noqa: F401
from pachyderm.typing_helpers import Hist
from pachyderm import histogram

if TYPE_CHECKING:
    from reaction_plane_fit import fit

logger = logging.getLogger(__name__)

# Type helpers
InputData = Dict[str, Dict[str, Union[Hist, histogram.Histogram1D]]]
Data = Dict["FitType", histogram.Histogram1D]

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
                                            component: "fit.FitComponent") -> fit_base.BaseFitResult:
    """ Create a component fit result from the fit component and the RP fit result.

    Args:
        fit_result: Fit result from the RP fit.
        component: Fit component for this fit result.
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

    return fit_base.BaseFitResult(
        parameters = parameters,
        free_parameters = free_parameters,
        fixed_parameters = fixed_parameters,
        values_at_minimum = values_at_minimum,
        errors_on_parameters = errors_on_parameters,
        covariance_matrix = covariance_matrix,
        # This will be determined and set later.
        errors = np.array([]),
    )

def format_input_data(data: Union[InputData, Data]) -> Data:
    """ Convert input data into a more convenient format.

    By using ``FitType``, we can very easily check that all fit components have the appropriate data.

    Args:
        data: Input data to be formatted.
    Returns:
        Properly formatted data, with ``FitType`` keys and histograms as values.
    """
    # Check if it's already properly formatted.
    formatted_data: Data = {}
    properly_formatted = all(isinstance(k, FitType) for k in data.keys())
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
                    formatted_data[FitType(region = region, orientation = rp_orientation)] = hist
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
