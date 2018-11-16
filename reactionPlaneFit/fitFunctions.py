#!/usr/bin/env python

""" Implement the underlying fit functions.

These functions are called repeatedly  for each value in an array by iminuit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
from numpy import sin, cos
import probfit

logger = logging.getLogger(__name__)

def signalWrapper(x, nsAmplitude, asAmpltiude, nsSigma, asSigma, signalPedestal, **kwargs):
    """ Wrapper for minuit that basically reassigns descriptive parameter names to shorter names
    to make the function definition less verbose.

    Args:
        x (float): Delta phi value for which the signal will be calculated.
        nsAmplitude (float): Near-side gaussian amplitude.
        asAmpltiude (float): Away-side gaussian amplitude.
        nsSigma (float): Near-side gaussian sigma.
        asSigma (float): Away-side guassian sigma.
        pedestal (float): Pedestal on which the signal sits.
        kwargs (dict): Used to absorbs extra possible parameters from minuit.
    Returns:
        float: Value calculated by the function.
    """
    return signal(x = x, A1 = nsAmplitude, A2 = asAmpltiude, s1 = nsSigma, s2 = asSigma, pedestal = signalPedestal)

def signal(x, A1, A2, s1, s2, pedestal):
    r""" Function for fitting the signal region of the reaction plane fit.

    The signal function consists of two gaussian peaks (one at the NS (0.0) and
    one at the AS (np.pi)), along with a pedestal. Gaussians are normalized and
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

def determine_signal_dominated_fit_function(rpOrientation, resolutionParameters, reactionPlaneParameters):
    """ Determine the signal fit function.

    This function consists of near-side and away side gaussians representating the
    signal added to a function to describe the background. For inclusive EP
    orientations, this is a Fourier series, while for other ep orientations, this is
    a RPF function.

    Args:
        rpOrientation (eventPlaneOrientation): The event plane orientation.
        resolutionParameters (dict): Maps resolution paramaeters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82". Only used for RPF.
        reactionPlaneParameters (tuple):
    Returns:
        function: The background function
    """
    # Signal function
    signalFunc = signalWrapper
    # Background function
    backgroundFunc = determine_background_fit_function(rpOrientation, resolutionParameters, reactionPlaneParameters)

    if rpOrientation == "all":
        # We don't need to rename the all angles function because we can only use
        # the signal fit on all angles alone. If we fit the other event plane angles
        # at the same time, it will double count
        signalDominatedFunc = probfit.functor.AddPdf(signalFunc, backgroundFunc)
    else:
        # Rename the variables so each signal related variable is independent for each EP
        # We do this by renaming all parameters that are _not_ used in the background
        # NOTE: prefixSkipParameters includes the variable "x", but that is fine, as it
        #       would be automatically excluded (and we don't want to prefix it anyway)!
        prefixSkipParameters = probfit.describe(backgroundFunc)

        # Sum the functions together
        # NOTE: The "BG" prefix shouldn't ever need to be used, but it is included so that
        #       it fails clearly in the case that a mistake is made and the prefix is actually
        #       matched to and applied to some paramter
        signalDominatedFunc = probfit.functor.AddPdf(signalFunc, backgroundFunc, prefix = [rpOrientation + "_", "BG"], skip_prefix = prefixSkipParameters)

    logger.debug(f"rpOrientation: {rpOrientation}, signalDominatedFunc: {probfit.describe(signalDominatedFunc)}")
    return signalDominatedFunc

def background_wrapper(phi, c, resolutionParameters):
    """ Wrapper around the RPF background function to allow the specification of
    relevant parameters.

    Args:
        phi (float): Center of tbe event plane bin. Matches up to phi_s in the RPF paper.
        c (float): Width of the event plane bin. Matches up to c in the RPF paper.
        resolutionParameters (dict): Maps resolution paramaeters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82".
    Returns:
        function: Wrapper around the actual background function with the specified parameters.
    """
    def bg_wrapper(x, B, v2_t, v2_a, v4_t, v4_a, v1, v3, **kwargs):
        """ Defines the background function that will be passed to a particular cost function
        (and eventually, to iminuit).

        NOTE: The arguments must be specified explicitly here because minuit uses the argument
              names to deteremine which arguments should be passed to this function via
              `iminuit.util.describe()`.

        Args:
            x (float): Delta phi value for which the background will be calculated.
            B (float): Overall multiplicative background level.
            v2_t (float): Trigger v_{2}.
            v2_a (float): Associated v_{2}.
            v4_t (float): Trigger v_{4}.
            v4_a (float): Associated v_{4}
            v1 (float): v1 parameter.
            v3 (float): v3 parameter.
            kwargs (dict): Used to absorbs extra possible parameters from minuit.
        Returns:
            float: Value calculated by the function.
        """
        # The resolution parameters are passed directly instead of via the backgroundParameters because
        # they are not something that should vary in our fit
        return background(x, phi = phi, c = c, resolutionParameters = resolutionParameters,
                          B = B,
                          v2_t = v2_t, v2_a = v2_a,
                          v4_t = v4_t, v4_a = v4_a,
                          v1 = v1,
                          v3 = v3)

    return bg_wrapper

def background(x, phi, c, resolutionParameters, B, v2_t, v2_a, v4_t, v4_a, v1, v3, **kwargs):
    """ The background function is of the form specified in the RPF paper.

    Resolution parameters implemented include R{2,2} through R{8,2}, which denotes the resolution of an order
    m reaction plane with respect to n = 2 reaction plane. R{8,2} is the highest value which should contribute
    to v_4^{eff}.

    Args:
        x (float): Delta phi value for which the background will be calculated.
        phi (float): Center of tbe event plane bin. Matches up to phi_s in the RPF paper
        c (float): Width of the event plane bin. Matches up to c in the RPF paper
        resolutionParameters (dict): Contains the resolution parameters with respect to the n = 2 reaction plane.
            Note the information about the parameters above. The expected keys are "R22" - "R82".
        B (float): Overall multiplicative background level.
        v2_t (float): Trigger v_{2}.
        v2_a (float): Associated v_{2}.
        v4_t (float): Trigger v_{4}.
        v4_a (float): Associated v_{4}
        v1 (float): v1 parameter.
        v3 (float): v3 parameter.
        kwargs (dict): Used to absorbs extra possible parameters from minuit.
    Returns:
        float: Values calculated by the function.
    """
    # Define individual resolution variables to make the expressions more concise.
    R22 = resolutionParameters["R22"]
    R42 = resolutionParameters["R42"]
    R62 = resolutionParameters["R62"]
    R82 = resolutionParameters["R82"]

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
    return fourier(x, BR, v2R, v2_a, v4R, v4_a, v1, v3)

def fourier(x, BG, v2_t, v2_a, v4_t, v4_a, v1, v3, **kwargs):
    """ Fourier decomposition for use in describing the background.

    NOTE: `B` was renamed to `BG` in the args so the argument would be decoupled (ie separate)
          from the background of the RPF! This is needed because iminuit routes the arguments
          based on these names.

    Args:
        BG (float): Overall multiplicative background level.
        v2_t (float): Trigger v_{2}.
        v2_a (float): Associated v_{2}.
        v4_t (float): Trigger v_{4}.
        v4_a (float): Associated v_{4}
        v1 (float): v1 parameter.
        v3 (float): v3 parameter.
        kwargs (dict): Used to absorbs extra possible parameters from minuit.
    Returns:
        float: Values calculated by the function.
    """
    return BG * (1 + 2 * v1 * np.cos(x)
                   + 2 * v2_t * v2_a * np.cos(2 * x)
                   + 2 * v3 * np.cos(3 * x)
                   + 2 * v4_t * v4_a * np.cos(4 * x))

def determine_background_fit_function(rpOrientation, resolutionParameters, reactionPlaneParameters):
    """ Determine the background fit function.

    For inclusive EP orientations, this is a Fourier series. For other EP orientations,
    it is an RPF function.

    Args:
        rpOrientation (eventPlaneOrientation): The event plane orientation.
        resolutionParameters (dict): Maps resolution paramaeters of the form "R22" (for
            the R_{2,2} parameter) to the value. Expects "R22" - "R82". Only used for RPF.
        reactionPlaneParameters (tuple):
    Returns:
        function: The background function
    """
    if rpOrientation == "all":
        backgroundFunc = fourier
    else:
        # Get RPF parameters
        (phiS, c) = reactionPlaneParameters
        # Define the function based on the parameters above.
        backgroundFunc = background_wrapper(phi = phiS[rpOrientation],
                                            c = c[rpOrientation],
                                            resolutionParameters = resolutionParameters)

    logger.debug(f"rpOrientation: {rpOrientation}, backgroundFunc: {probfit.describe(backgroundFunc)}")
    return backgroundFunc

