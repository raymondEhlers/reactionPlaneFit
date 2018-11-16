#!/usr/bin/env python

""" Tests for the various fit functions

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import numpy as np
import probfit
import pytest
import logging

import reactionPlaneFit.fitFunctions as fitFunctions

logger = logging.getLogger(__name__)

def setupValues():
    """ Setup the required binning for tests. """
    # Use 50 bins
    # NOTE: These are evaluated on the edges
    # TODO: Should they be offset by bin width??
    return np.linspace(-1. / 2 * np.pi, 3. / 2 * np.pi, 51)

@pytest.fixture
def setupSignal():
    """ Setup for testing the signal function. """
    values = setupValues()
    nsAmplitude = 1
    asAmpltiude = 0.5
    nsSigma = 0.2
    asSigma = 0.7
    signalPedestal = 10
    # These will be ignored.
    kwargs = {"randomValue": "toBeIgnored"}

    def testWrapper(x):
        """ Trivial wraper so we can set the parameter values in the fixture. """
        return fitFunctions.signalWrapper(x, nsAmplitude = nsAmplitude,
                                          asAmpltiude = asAmpltiude,
                                          nsSigma = nsSigma,
                                          asSigma = asSigma,
                                          signalPedestal = signalPedestal,
                                          **kwargs)

    expected = np.array([10.        , 10.        , 10.        , 10.00000004, 10.00000128,  # noqa: E203
                         10.00003006, 10.0004764 , 10.00508941, 10.03663746, 10.17771922,  # noqa: E203
                         10.58088721, 11.27937272, 11.89867369, 11.89868362, 11.27940816,
                         10.58096742, 10.17788456, 10.0369637 , 10.00571129, 10.00162372,  # noqa: E203
                         10.00207944, 10.00354581, 10.00593608, 10.00962585, 10.01511418,
                         10.02297917, 10.03382891, 10.04822202, 10.06655898, 10.0889553 ,  # noqa: E203
                         10.11511738, 10.14424932, 10.17502116, 10.20562278, 10.23391374,
                         10.25765829, 10.27481244, 10.28381315, 10.28381315, 10.27481244,
                         10.25765829, 10.23391374, 10.20562278, 10.17502116, 10.14424932,
                         10.11511738, 10.0889553 , 10.06655898, 10.04822202, 10.03382891,  # noqa: E203
                         10.02297917])

    return (values, testWrapper, expected)

@pytest.fixture
def setupBackground():
    """ Setup for testing the background functions. """
    values = setupValues()

    phi = 0
    c = np.pi / 6
    resolutionParameters = {"R22": 0.5, "R42": 0.4, "R62": 0.1, "R82": 0.1}
    B = 10
    v2_t = 0.05
    v2_a = 0.05
    v4_t = 0.025
    v4_a = 0.025
    v1 = 0.1
    v3 = 0.02
    # These will be ignored.
    kwargs = {"randomValue": "toBeIgnored"}

    def testWrapper(x):
        """ Trivial wrapper so we can set the parameter values in the fixture. Note call a wrapper and then
        return that function with the proper parameters set. """
        func = fitFunctions.background_wrapper(phi = phi, c = c, resolutionParameters = resolutionParameters)
        return func(x = x,
                    B = B,
                    v2_t = v2_t, v2_a = v2_a,
                    v4_t = v4_t, v4_a = v4_a,
                    v1 = v1,
                    v3 = v3,
                    **kwargs)

    expected = np.array([3.37312343, 3.41001413, 3.45492997, 3.5147532, 3.59511442,
                         3.69888285, 3.82500203, 3.96795878, 4.11806146, 4.26254461,
                         4.38734488, 4.47924491, 4.52799244, 4.52799244, 4.47924491,
                         4.38734488, 4.26254461, 4.11806146, 3.96795878, 3.82500203,
                         3.69888285, 3.59511442, 3.5147532 , 3.45492997, 3.41001413,  # noqa: E203
                         3.37312343, 3.33764871, 3.29849401, 3.25282532, 3.20024986,
                         3.14248097, 3.08264883, 3.02447321, 2.9715112 , 2.92663772,  # noqa: E203
                         2.89182788, 2.86821481, 2.85632043, 2.85632043, 2.86821481,
                         2.89182788, 2.92663772, 2.9715112 , 3.02447321, 3.08264883,  # noqa: E203
                         3.14248097, 3.20024986, 3.25282532, 3.29849401, 3.33764871,
                         3.37312343])

    return (values, testWrapper, expected)

@pytest.fixture
def setupFourier(loggingMixin):
    """ Setup for testing the fourier series function. """
    values = setupValues()

    BG = 10
    v2_t = 0.05
    v2_a = 0.05
    v4_t = 0.025
    v4_a = 0.025
    v1 = 0.1
    v3 = 0.02
    # These will be ignored.
    kwargs = {"randomValue": "toBeIgnored"}

    def testWrapper(x):
        """ Trivial wraper so we can set the parameter values in the fixture. """
        return fitFunctions.fourier(x = x,
                                    BG = BG,
                                    v2_t = v2_t, v2_a = v2_a,
                                    v4_t = v4_t, v4_a = v4_a,
                                    v1 = v1,
                                    v3 = v3,
                                    **kwargs)

    expected = np.array([9.9625     , 10.06594132, 10.18644343, 10.33865473, 10.53218308,  # noqa: E203, E241
                         10.76958434, 11.04534796, 11.34607188, 11.65184372, 11.93865899,
                         12.1815407 , 12.3579066 , 12.45068138, 12.45068138, 12.3579066 ,  # noqa: E203, E241
                         12.1815407 , 11.93865899, 11.65184372, 11.34607188, 11.04534796,  # noqa: E203, E241
                         10.76958434, 10.53218308, 10.33865473, 10.18644343, 10.06594132,
                         9.9625     ,  9.85910803,  9.73932157,  9.59001817,  9.40358976,  # noqa: E203, E241
                         9.17928854 ,  8.92357013,  8.64942184,  8.37479861,  8.12039887,  # noqa: E203, E241
                         7.90708643 ,  7.75329526,  7.67274467,  7.67274467,  7.75329526,  # noqa: E203, E241
                         7.90708643 ,  8.12039887,  8.37479861,  8.64942184,  8.92357013,  # noqa: E203, E241
                         9.17928854 ,  9.40358976,  9.59001817,  9.73932157,  9.85910803,  # noqa: E203, E241
                         9.9625])  # noqa: E203

    return (values, testWrapper, expected)

@pytest.mark.parametrize("setupFit", [
    "setupSignal",
    "setupBackground",
    "setupFourier"
], ids = ["Signal", "Background", "Fourier"])
def testFitFunctions(loggingMixin, setupFit, request):
    """ Test the fit functions. Each `setupFit` value refers to a different fixture. """
    values, func, expected = request.getfixturevalue(setupFit)
    output = np.zeros(len(values))
    for i, val in enumerate(values):
        output[i] = func(x = val)

    # NOTE: This calls assert internally
    np.testing.assert_allclose(output, expected)

def testSignalArgs(loggingMixin):
    """ Test the arguments for the signal function. """
    assert probfit.describe(fitFunctions.signalWrapper) == ["x",
                                                            "nsAmplitude", "asAmpltiude",
                                                            "nsSigma", "asSigma",
                                                            "signalPedestal"]

def testRPFBackgroundArgs(loggingMixin):
    """ Test the arguments for the RPF function. """
    phi = 0
    c = np.pi / 6
    resolutionParameters = {"R22": 0.5, "R42": 0.4, "R62": 0.1, "R82": 0.1}
    func = fitFunctions.background_wrapper(phi = phi, c = c, resolutionParameters = resolutionParameters)
    assert probfit.describe(func, verbose = True) == ["x",
                                                      "B",
                                                      "v2_t", "v2_a",
                                                      "v4_t", "v4_a",
                                                      "v1", "v3"]

def testFourierArgs(loggingMixin):
    """ Test the arguments for the Fourier series function. """
    assert probfit.describe(fitFunctions.fourier) == ["x",
                                                      "BG",
                                                      "v2_t", "v2_a",
                                                      "v4_t", "v4_a",
                                                      "v1", "v3"]

