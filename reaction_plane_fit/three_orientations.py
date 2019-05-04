#!/usr/bin/env python

""" Implements the three RP orientation fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
from numpy import sin, cos
from typing import Any, Dict

from reaction_plane_fit import base
from reaction_plane_fit import fit
from reaction_plane_fit import functions

logger = logging.getLogger(__name__)

# Define the relevant fit components for this set of RP orientation.
class SignalFitComponent(fit.SignalFitComponent):
    """ Signal fit component for three RP orientations. """
    def determine_fit_function(self, resolution_parameters: fit.ResolutionParameters, reaction_plane_parameter: base.ReactionPlaneParameter) -> None:
        self.fit_function = functions.determine_signal_dominated_fit_function(
            rp_orientation = self.rp_orientation,
            resolution_parameters = resolution_parameters,
            reaction_plane_parameter = reaction_plane_parameter,
            rp_background_function = background,
        )
        self.background_function = functions.determine_background_fit_function(
            rp_orientation = self.rp_orientation,
            resolution_parameters = resolution_parameters,
            reaction_plane_parameter = reaction_plane_parameter,
            rp_background_function = background,
        )

class BackgroundFitComponent(fit.BackgroundFitComponent):
    """ Background fit component for three RP orientations. """
    def determine_fit_function(self, resolution_parameters: fit.ResolutionParameters, reaction_plane_parameter: base.ReactionPlaneParameter) -> None:
        self.fit_function = functions.determine_background_fit_function(
            rp_orientation = self.rp_orientation,
            resolution_parameters = resolution_parameters,
            reaction_plane_parameter = reaction_plane_parameter,
            rp_background_function = background,
        )
        # This is identical to the fit function.
        self.background_function = self.fit_function

class ReactionPlaneFit(fit.ReactionPlaneFit):
    """ Base class for reaction plane fit for 3 reaction plane orientations.

    Attributes:
        _rp_orientations: List of the reaction plane orientations.
        reaction_plane_parameter: Reaction plane parameters, including the orientation, center, and width.
    """
    _rp_orientations = ["in_plane", "mid_plane", "out_of_plane", "inclusive"]
    reaction_plane_parameters = {
        "in_plane": base.ReactionPlaneParameter(orientation = "in_plane",
                                                phiS = 0,
                                                c = np.pi / 6.),
        # NOTE: This c value is halved in the fit to account for the four non-continuous regions
        "mid_plane": base.ReactionPlaneParameter(orientation = "mid_plane",
                                                 phiS = np.pi / 4.,
                                                 c = np.pi / 12.),
        "out_of_plane": base.ReactionPlaneParameter(orientation = "out_of_plane",
                                                    phiS = np.pi / 2.,
                                                    c = np.pi / 6.),
        # This is really meaningful, so it should be ignored.
        # However, it is helpful for it to be defined in this dict for lookup purposes.
        "inclusive": base.ReactionPlaneParameter(orientation = "inclusive",
                                                 phiS = 0,
                                                 c = np.pi / 2),
    }

class BackgroundFit(ReactionPlaneFit):
    """ RPF for background region in 3 reaction plane orientations.

    This is a simple helper class to define the necessary fit component. Contains fit components for
    3 background RP orientations.

    Args:
        Same as for ``ReactionPlaneFit``.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        # Create the base class first
        super().__init__(*args, **kwargs)

        # Setup the fit components
        for orientation in self.rp_orientations:
            fit_type = base.FitType(region = "background", orientation = orientation)
            self.components[fit_type] = BackgroundFitComponent(rp_orientation = fit_type.orientation,
                                                               resolution_parameters = self.resolution_parameters,
                                                               use_log_likelihood = self.use_log_likelihood)

        # Complete basic setup of the components by setting up the fit functions.
        self._setup_component_fit_functions()

class InclusiveSignalFit(ReactionPlaneFit):
    """ RPF for inclusive signal region, and background region in 3 reaction planes orientations.

    This is a simple helper class to define the necessary fit component. Contains an inclusive signal fit,
    and 3 background RP orientations.

    Args:
        Same as for ``ReactionPlaneFit``.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        # Create the base class first
        super().__init__(*args, **kwargs)

        # Setup the fit components
        fit_type = base.FitType(region = "signal", orientation = "inclusive")
        self.components[fit_type] = SignalFitComponent(rp_orientation = fit_type.orientation,
                                                       resolution_parameters = self.resolution_parameters,
                                                       use_log_likelihood = self.use_log_likelihood)
        for orientation in self.rp_orientations:
            fit_type = base.FitType(region = "background", orientation = orientation)
            self.components[fit_type] = BackgroundFitComponent(rp_orientation = fit_type.orientation,
                                                               resolution_parameters = self.resolution_parameters,
                                                               use_log_likelihood = self.use_log_likelihood)

        # Complete basic setup of the components by setting up the fit functions.
        self._setup_component_fit_functions()

class SignalFit(ReactionPlaneFit):
    """ RPF for signal and background regions with 3 reaction plane orientations.

    This is a simple helper class to define the necessary fit component. Contains 3 signal
    orientations and 3 background RP orientations.

    Args:
        Same as for ``ReactionPlaneFit``.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        # Create the base class first
        super().__init__(*args, **kwargs)

        # Setup the fit components
        for region, fit_component in [("signal", SignalFitComponent), ("background", BackgroundFitComponent)]:
            # Help out mypy
            assert isinstance(fit_component, (SignalFitComponent, BackgroundFitComponent))
            for orientation in self.rp_orientations:
                fit_type = base.FitType(region = region, orientation = orientation)
                self.components[fit_type] = fit_component(rp_orientation = fit_type.orientation,
                                                          resolution_parameters = self.resolution_parameters,
                                                          use_log_likelihood = self.use_log_likelihood)

        # Complete basic setup of the components by setting up the fit functions.
        self._setup_component_fit_functions()

def background(x: float, phi: float, c: float, resolution_parameters: Dict[str, float], B: float, v2_t: float, v2_a: float, v4_t: float, v4_a: float, v1: float, v3: float, **kwargs: float) -> float:
    """ The background function is of the form specified in the RPF paper.

    Resolution parameters implemented include R{2,2} through R{8,2}, which denotes the resolution of an order
    m reaction plane with respect to n = 2 reaction plane. R{8,2} is the highest value which should contribute
    to v_4^{eff}.

    Args:
        x (float): Delta phi value for which the background will be calculated.
        phi (float): Center of the reaction plane bin. Matches up to phi_s in the RPF paper
        c (float): Width of the reaction plane bin. Matches up to c in the RPF paper
        resolution_parameters (dict): Contains the resolution parameters with respect to the n = 2 reaction plane.
            Note the information about the parameters above. The expected keys are "R22" - "R82".
        B (float): Overall multiplicative background level.
        v2_t (float): Trigger v_{2}.
        v2_a (float): Associated v_{2}.
        v4_t (float): Trigger v_{4}.
        v4_a (float): Associated v_{4}
        v1 (float): v1 parameter.
        v3 (float): v3 parameter.
        kwargs (dict): Used to absorbs extra possible parameters from Minuit (especially when used in
                conjunction with other functions).
    Returns:
        float: Values calculated by the function.
    """
    # Define individual resolution variables to make the expressions more concise.
    R22 = resolution_parameters["R22"]
    R42 = resolution_parameters["R42"]
    R62 = resolution_parameters["R62"]
    R82 = resolution_parameters["R82"]

    num = v2_t + cos(2 * phi) * sin(2 * c) / (2 * c) * R22 \
        + v4_t * cos(2 * phi) * sin(2 * c) / (2 * c) * R22 \
        + v2_t * cos(4 * phi) * sin(4 * c) / (4 * c) * R42 \
        + v4_t * cos(6 * phi) * sin(6 * c) / (6 * c) * R62
    den = 1 \
        + 2 * v2_t * cos(2 * phi) * sin(2 * c) / (2 * c) * R22 \
        + 2 * v4_t * cos(4 * phi) * sin(4 * c) / (4 * c) * R42
    v2R = num / den
    num2 = v4_t + cos(4 * phi) * sin(4 * c) / (4 * c) * R42 \
        + v2_t * cos(2 * phi) * sin(2 * c) / (2 * c) * R22 \
        + v2_t * cos(6 * phi) * sin(6 * c) / (6 * c) * R62 \
        + v4_t * cos(8 * phi) * sin(8 * c) / (8 * c) * R82
    v4R = num2 / den
    BR = B * den * c * 2 / np.pi
    factor = 1.0
    # In the case of mid-plane, it has 4 regions instead of 2
    if c == np.pi / 12.0:
        factor = 2.0
    BR = BR * factor
    return functions.fourier(x, BR, v2R, v2_a, v4R, v4_a, v1, v3)

