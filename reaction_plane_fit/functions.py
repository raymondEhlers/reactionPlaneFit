#!/usr/bin/env python

""" Based functions shared by all fit implementations.

The fit functions (ie all functions defined here except those to ``determine_...(...)`` functions) are called repeatedly
for each value in an array by ``iminuit``.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import probfit
from typing import Callable

logger = logging.getLogger(__name__)

def determine_signal_dominated_fit_function(rp_orientation: str, resolution_parameters: dict, reaction_plane_parameter, rp_background_function: Callable[..., float]) -> Callable[..., float]:
    """ Determine the signal fit function.

    This function consists of near-side and away side gaussians representing the signal added to a function
    to describe the background. For inclusive RP orientations, this is a Fourier series, while for other RP
    orientations, this is a RPF function.

    Args:
        rp_orientation (str): The reaction plane orientation.
        resolution_parameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82". Only used for RPF.
        reaction_plane_parameter (base.ReactionPlaneParameter): Reaction plane parameters for the selected
                reaction plane.
        rp_background_function (function): Background RP fit function.
    Returns:
        Callable: The signal fit function.
    """
    # Signal function
    signal_func = signal_wrapper
    # Background function
    background_func = determine_background_fit_function(rp_orientation,
                                                        resolution_parameters,
                                                        reaction_plane_parameter,
                                                        rp_background_function = rp_background_function)

    if rp_orientation == "inclusive":
        # We don't need to rename the all orientations function because we can only use
        # the signal fit on all orientations alone. If we fit the other reaction plane orientations
        # at the same time, it will double count.
        signal_dominated_func = probfit.functor.AddPdf(signal_func, background_func)
    else:
        # Rename the variables so each signal related variable is independent for each RP
        # We do this by renaming all parameters that are _not_ used in the background
        # NOTE: prefix_skip_parameters includes the variable "x", but that is fine, as it
        #       would be automatically excluded (and we don't want to prefix it anyway)!
        prefix_skip_parameters = probfit.describe(background_func)

        # Sum the functions together
        # NOTE: The "BG" prefix shouldn't ever need to be used, but it is included so that
        #       it fails clearly in the case that a mistake is made and the prefix is actually
        #       matched to and applied to some parameter.
        signal_dominated_func = probfit.functor.AddPdf(signal_func, background_func, prefix = [rp_orientation + "_", "BG"], skip_prefix = prefix_skip_parameters)

    logger.debug(f"rp_orientation: {rp_orientation}, signal_dominated_func: {probfit.describe(signal_dominated_func)}")
    return signal_dominated_func

def determine_background_fit_function(rp_orientation: str, resolution_parameters: dict, reaction_plane_parameter, rp_background_function: Callable[..., float]) -> Callable[..., float]:
    """ Determine the background fit function.

    For inclusive RP orientations, this is a Fourier series. For other RP orientations,
    it is an RPF function.

    Note:
        If the RP orientation is inclusive, it is assumed to be labeled as "inclusive".

    Args:
        rp_orientation (str): The reaction plane orientation.
        resolution_parameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82". Only used for RPF.
        reaction_plane_parameter (base.ReactionPlaneParameter): Reaction plane parameters for the selected
                reaction plane.
        rp_background_function (function): Background RP fit function.
    Returns:
        function: The background function.
    """
    if rp_orientation == "inclusive":
        background_func = fourier
    else:
        # Define the function based on the parameters passed to the function.
        background_func = background_wrapper(phi = reaction_plane_parameter.phiS,
                                             c = reaction_plane_parameter.c,
                                             resolution_parameters = resolution_parameters,
                                             background_function = rp_background_function)

    logger.debug(f"rp_orientation: {rp_orientation}, background_func: {probfit.describe(background_func)}")
    return background_func

def signal_wrapper(x: float, ns_amplitude: float, as_amplitude: float, ns_sigma: float, as_sigma: float, signal_pedestal: float, **kwargs: dict) -> float:
    """ Wrapper for use with ``iminuit`` that basically reassigns descriptive parameter names to shorter names.

    These shorter names make the function definition less verbose to look and therefore easier to understand.

    Args:
        x (float): Delta phi value for which the signal will be calculated.
        ns_amplitude (float): Near-side gaussian amplitude.
        as_amplitude (float): Away-side gaussian amplitude.
        ns_sigma (float): Near-side gaussian sigma.
        as_sigma (float): Away-side gaussian sigma.
        signal_pedestal (float): Pedestal on which the signal sits.
        kwargs (dict): Used to absorbs extra possible parameters from Minuit (especially when used in
                conjunction with other functions).
    Returns:
        float: Value calculated by the function.
    """
    return signal(x = x, A1 = ns_amplitude, A2 = as_amplitude, s1 = ns_sigma, s2 = as_sigma, pedestal = signal_pedestal)

def signal(x: float, A1: float, A2: float, s1: float, s2: float, pedestal: float) -> float:
    r""" Function for fitting the signal region of the reaction plane fit.

    The signal function consists of two gaussian peaks (one at the NS (0.0) and
    one at the AS (``np.pi``)), along with a pedestal. Gaussians are normalized and
    are of the form:

    .. math:: 1/(\sigma*\sqrt(2*\pi) * e^{-(x-0.0)^{2}/(2*\sigma^{2})}

    Args:
        x (float): Delta phi value for which the signal will be calculated.
        A1 (float): Near-side gaussian amplitude.
        A2 (float): Away-side gaussian amplitude.
        s1 (float): Near-side gaussian sigma.
        s2 (float): Away-side gaussian sigma.
        pedestal (float): Pedestal on which the signal sits.
    Returns:
        float: Value calculated by the function.
    """
    return A1 * probfit.pdf.gaussian(x = x, mean = 0.0, sigma = s1) \
        + A2 * probfit.pdf.gaussian(x = x, mean = np.pi, sigma = s2) \
        + pedestal

def background_wrapper(phi: float, c: float, resolution_parameters: dict, background_function: Callable[..., float]):
    """ Wrapper around the RPF background function to allow the specification of relevant parameters.

    This allows the more standard background function to be used without having to pass these fixed parameters
    to the function every time.

    Args:
        phi (float): Center of the reaction plane bin. Matches up to phi_s in the RPF paper.
        c (float): Width of the reaction plane bin. Matches up to c in the RPF paper.
        resolution_parameters (dict): Maps resolution parameters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82".
        background_function (Callable): Background function to be wrapped.
    Returns:
        function: Wrapper around the actual background function with the specified parameters.
    """
    def bg_wrapper(x: float, B: float, v2_t: float, v2_a: float, v4_t: float, v4_a: float, v1: float, v3: float, **kwargs: dict) -> float:
        """ Defines the background function that will be passed to a particular cost function
        (and eventually, to ``iminuit``).

        Note:
            The arguments must be specified explicitly here because Minuit uses the argument names to
            determine which arguments should be passed to this function via ``iminuit.util.describe()``.

        Args:
            x (float): Delta phi value for which the background will be calculated.
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
            float: Value calculated by the function.
        """
        # The resolution parameters are passed directly instead of via the backgroundParameters because
        # they are not something that should vary in our fit
        return background_function(x, phi = phi, c = c, resolution_parameters = resolution_parameters,
                                   B = B,
                                   v2_t = v2_t, v2_a = v2_a,
                                   v4_t = v4_t, v4_a = v4_a,
                                   v1 = v1,
                                   v3 = v3)

    return bg_wrapper

def fourier(x: float, BG: float, v2_t: float, v2_a: float, v4_t: float, v4_a: float, v1: float, v3: float, **kwargs: dict) -> float:
    """ Fourier decomposition for use in describing the background.

    Note:
        `B` was renamed to `BG` in the args so the argument would be decoupled (ie separate) from the background
        of the RPF! This is needed because ``iminuit`` routes the arguments based on these names.

    Args:
        BG (float): Overall multiplicative background level.
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
    return BG * (1 + 2 * v1 * np.cos(x)
                   + 2 * v2_t * v2_a * np.cos(2 * x)
                   + 2 * v3 * np.cos(3 * x)
                   + 2 * v4_t * v4_a * np.cos(4 * x))

