#!/usr/bin/env python

""" Tests for the various fit functions.

Including the signal (Gaussian) and fourier functions, as well as the three orientation background function.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import iminuit
import numpy as np
import pytest
import logging

from reaction_plane_fit import functions
from reaction_plane_fit import three_orientations

logger = logging.getLogger(__name__)

def setup_values():
    """ Setup the required binning for tests. """
    # Use 50 bins evaluted at the bin centers
    edges = np.linspace(-1. / 2 * np.pi, 3. / 2 * np.pi, 51)
    return (edges[1:] + edges[:-1]) / 2

@pytest.fixture
def setup_signal():
    """ Setup for testing the signal function. """
    values = setup_values()
    ns_amplitude = 1
    as_amplitude = 0.5
    ns_sigma = 0.2
    as_sigma = 0.7
    signal_pedestal = 10
    # These will be ignored.
    kwargs = {"randomValue": "toBeIgnored"}

    def test_wrapper(x):
        """ Trivial wraper so we can set the parameter values in the fixture. """
        return functions.signal_wrapper(x, ns_amplitude = ns_amplitude,
                                        as_amplitude = as_amplitude,
                                        ns_sigma = ns_sigma,
                                        as_sigma = as_sigma,
                                        signal_pedestal = signal_pedestal,
                                        **kwargs)

    expected = np.array([10.        , 10.        , 10.00000001, 10.00000023, 10.00000651,  # noqa: E203
                         10.00012571, 10.00163587, 10.01434588, 10.08477372, 10.33755505,
                         10.90568249, 11.63740149, 11.99472345, 11.63742273, 10.90573684,
                         10.337671  , 10.08500704, 10.01479817, 10.00248398, 10.00166529,  # noqa: E203
                         10.00271259, 10.00460573, 10.00758958, 10.01211048, 10.01871152,
                         10.02799372, 10.04055236, 10.05688209, 10.07725717, 10.10160285,
                         10.12938289, 10.15953343, 10.19047168, 10.22019781, 10.24649008,
                         10.26717128, 10.28040385, 10.28495877, 10.28040385, 10.26717128,
                         10.24649008, 10.22019781, 10.19047168, 10.15953343, 10.12938289,
                         10.10160285, 10.07725717, 10.05688209, 10.04055236, 10.02799372])

    return (values, test_wrapper, expected)

@pytest.fixture
def setup_three_orientations_background():
    """ Setup for testing the background functions. """
    values = setup_values()

    phi = 0
    c = np.pi / 6
    resolution_parameters = {"R22": 0.5, "R42": 0.4, "R62": 0.1, "R82": 0.1}
    B = 10
    v2_t = 0.05
    v2_a = 0.05
    v4_t = 0.025
    v4_a = 0.025
    v1 = 0.1
    v3 = 0.02
    # These will be ignored.
    kwargs = {"randomValue": "toBeIgnored"}

    def test_wrapper(x):
        """ Trivial wrapper so we can set the parameter values in the fixture. Note call a wrapper and then
        return that function with the proper parameters set. """
        func = functions.background_wrapper(phi = phi, c = c, resolution_parameters = resolution_parameters,
                                            background_function = three_orientations.background)
        return func(x = x,
                    B = B,
                    v2_t = v2_t, v2_a = v2_a,
                    v4_t = v4_t, v4_a = v4_a,
                    v1 = v1,
                    v3 = v3,
                    **kwargs)

    expected = np.array([3.39100167, 3.43102803, 3.48257833, 3.55210639, 3.64403686,
                         3.75938779, 3.89489261, 4.04286082, 4.19187824, 4.32827699,
                         4.43814052, 4.50948689, 4.53422179, 4.50948689, 4.43814052,
                         4.32827699, 4.19187824, 4.04286082, 3.89489261, 3.75938779,
                         3.64403686, 3.55210639, 3.48257833, 3.43102803, 3.39100167,
                         3.35557455, 3.31876638, 3.27655397, 3.22733637, 3.17184259,
                         3.11259449, 3.05312032, 2.99714077, 2.94791838, 2.90788648,
                         2.87857809, 2.8607864 , 2.85483043, 2.8607864 , 2.87857809,  # noqa: E203
                         2.90788648, 2.94791838, 2.99714077, 3.05312032, 3.11259449,
                         3.17184259, 3.22733637, 3.27655397, 3.31876638, 3.35557455])

    return (values, test_wrapper, expected)

@pytest.fixture
def setup_fourier(logging_mixin):
    """ Setup for testing the fourier series function. """
    values = setup_values()

    BG = 10
    v2_t = 0.05
    v2_a = 0.05
    v4_t = 0.025
    v4_a = 0.025
    v1 = 0.1
    v3 = 0.02
    # These will be ignored.
    kwargs = {"randomValue": "toBeIgnored"}

    def test_wrapper(x):
        """ Trivial wraper so we can set the parameter values in the fixture. """
        return functions.fourier(x = x,
                                 BG = BG,
                                 v2_t = v2_t, v2_a = v2_a,
                                 v4_t = v4_t, v4_a = v4_a,
                                 v1 = v1,
                                 v3 = v3,
                                 **kwargs)

    expected = np.array([10.01313007, 10.12305519, 10.25783905, 10.42991185, 10.64555094,  # noqa: E203
                         10.90333404, 11.19370575, 11.49976533, 11.79919867, 12.067098  ,  # noqa: E203
                         12.27926694, 12.41552299, 12.4625    , 12.41552299, 12.27926694,  # noqa: E203
                         12.067098  , 11.79919867, 11.49976533, 11.19370575, 10.90333404,  # noqa: E203
                         10.64555094, 10.42991185, 10.25783905, 10.12305519, 10.01313007,
                          9.91187304,  9.80219137,  9.66898467,  9.50166122,  9.29593553,  # noqa: E241
                          9.05468342,  8.78777043,  8.51091095,  8.24373953,  8.00736862,  # noqa: E241
                          7.8217594 ,  7.70324299,  7.6625    ,  7.70324299,  7.8217594 ,  # noqa: E203, E241
                          8.00736862,  8.24373953,  8.51091095,  8.78777043,  9.05468342,  # noqa: E241
                          9.29593553,  9.50166122,  9.66898467,  9.80219137,  9.91187304])  # noqa: E241

    return (values, test_wrapper, expected)

@pytest.mark.parametrize("setup_fit", [
    "setup_signal",
    "setup_three_orientations_background",
    "setup_fourier"
], ids = ["Signal", "Three orientation background", "Fourier"])
def test_fit_functions(logging_mixin, setup_fit, request):
    """ Test the fit functions. Each `setup_fit` value refers to a different fixture. """
    values, func, expected = request.getfixturevalue(setup_fit)
    output = np.zeros(len(values))
    for i, val in enumerate(values):
        output[i] = func(x = val)

    assert np.allclose(output, expected)

def test_signal_args(logging_mixin):
    """ Test the arguments for the signal function. """
    assert iminuit.util.describe(functions.signal_wrapper) == [
        "x", "ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "signal_pedestal"
    ]

def test_three_orientations_background_args(logging_mixin, mocker):
    """ Test the arguments for the RPF function. """
    phi = 0
    c = np.pi / 6
    resolution_parameters = {"R22": 0.5, "R42": 0.4, "R62": 0.1, "R82": 0.1}
    func = functions.background_wrapper(phi = phi, c = c, resolution_parameters = resolution_parameters,
                                        background_function = mocker.MagicMock())
    assert iminuit.util.describe(func, verbose = True) == [
        "x", "B", "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3"
    ]

def test_fourier_args(logging_mixin):
    """ Test the arguments for the Fourier series function. """
    assert iminuit.util.describe(functions.fourier) == [
        "x", "BG", "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3"
    ]

