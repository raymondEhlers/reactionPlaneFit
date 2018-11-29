#!/usr/bin/env python

""" Integration tests for the inclusive signal and background only fits.

The tests are performed by leveraging the example fit code, as it runs an entire fit.

By using integration tests, we can tests large positions of the package more rapidly.
This of course trades granularity and may allow some branches to slip through, but the
time savings is worth it.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import numpy as np
import pkg_resources
import pytest

from reaction_plane_fit import base
from reaction_plane_fit import example

@pytest.fixture
def setup_integration_tests(loggingMixin):
    """ Setup shared expected values for the fit integration tests. """
    sample_data_filename = pkg_resources.resource_filename("reaction_plane_fit.sample_data", "three_orientations.root")

    return sample_data_filename

def compare_fit_result_to_expected(fit_result, expected_fit_result):
    """ Helper function to compare a fit result to an expected fit result.

    Args:
        fit_result (base.FitResult): The calculated fit result.
        expected_fit_result (base.FitResult): The expected fit result.
    Returns:
        bool: True if the fit results are the same.
    """
    assert fit_result.parameters == expected_fit_result.parameters
    assert fit_result.free_parameters == expected_fit_result.free_parameters
    assert fit_result.fixed_parameters == expected_fit_result.fixed_parameters
    assert np.isclose(fit_result.minimum_val, expected_fit_result.minimum_val)
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.values_at_minimum.keys()) == list(expected_fit_result.values_at_minimum.keys())
    # Need the extra tolerance to work on other systems.
    assert np.allclose(list(fit_result.values_at_minimum.values()), list(expected_fit_result.values_at_minimum.values()), rtol = 0.001)
    assert np.allclose(fit_result.x, expected_fit_result.x)
    assert fit_result.n_fit_data_points == expected_fit_result.n_fit_data_points
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.covariance_matrix.keys()) == list(expected_fit_result.covariance_matrix.keys())
    # Need the extra tolerance to work on other systems.
    assert np.allclose(list(fit_result.covariance_matrix.values()), list(expected_fit_result.covariance_matrix.values()), rtol = 0.001)

    # Calculated values
    assert fit_result.nDOF == expected_fit_result.nDOF

    # If all assertions passed, then return True to indicate the success.
    return True

@pytest.mark.slow
def test_inclusive_signal_fit(setup_integration_tests):
    """ Integration test for the inclusive signal fit.

    This uses the sample data in the ``testFiles`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    # NOTE: They are not calculated independently, so there are most like regression tests.
    expected_fit_result = base.RPFitResult(
        parameters = ["ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "signal_pedestal", "BG", "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3", "B"],
        free_parameters = ["ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "BG", "v2_t", "v2_a", "v4_t", "v4_a", "v3", "B"],
        fixed_parameters = ["signal_pedestal", "v1"],
        values_at_minimum = {
            'ns_amplitude': 1.7933561625700922, 'as_amplitude': 1.0001072229610664,
            'ns_sigma': 0.2531797052038343, 'as_sigma': 0.4304508803936734,
            'signal_pedestal': 0.0, 'BG': 24.51231187224201,
            'v2_t': 1.635883107545255e-05, 'v2_a': 0.04207445772474305,
            'v4_t': 0.0021469295947451894, 'v4_a': 0.002584005026287889,
            'v1': 0.0, 'v3': 0.0016802003999212695,
            'B': 74.09951310294166
        },
        covariance_matrix = {
            ('ns_amplitude', 'ns_amplitude'): 0.016484487745377686, ('ns_amplitude', 'as_amplitude'): 0.00937777016042199,
            ('ns_amplitude', 'ns_sigma'): 0.001433801183150422, ('ns_amplitude', 'as_sigma'): 0.004586628583904665,
            ('ns_amplitude', 'BG'): -0.004019509691337873, ('ns_amplitude', 'v2_t'): -3.5013665434579806e-05,
            ('ns_amplitude', 'v2_a'): 3.6570973251680785e-06, ('ns_amplitude', 'v4_t'): 3.047333679513599e-06,
            ('ns_amplitude', 'v4_a'): 1.0803267317539679e-06, ('ns_amplitude', 'v3'): -6.630890180366187e-05,
            ('ns_amplitude', 'B'): -0.0019040291327373684, ('as_amplitude', 'ns_amplitude'): 0.00937777016042199,
            ('as_amplitude', 'as_amplitude'): 0.025651478042855662, ('as_amplitude', 'ns_sigma'): 0.0008107414313640945,
            ('as_amplitude', 'as_sigma'): 0.008412717771853899, ('as_amplitude', 'BG'): -0.005534282484339136,
            ('as_amplitude', 'v2_t'): -3.973113571976044e-05, ('as_amplitude', 'v2_a'): 2.2158781868346477e-06,
            ('as_amplitude', 'v4_t'): 3.993982077267372e-06, ('as_amplitude', 'v4_a'): 2.0448291125666287e-06,
            ('as_amplitude', 'v3'): -4.486572540586001e-05, ('as_amplitude', 'B'): -0.0011687806471112946,
            ('ns_sigma', 'ns_amplitude'): 0.001433801183150422, ('ns_sigma', 'as_amplitude'): 0.0008107414313640945,
            ('ns_sigma', 'ns_sigma'): 0.00030892555738012, ('ns_sigma', 'as_sigma'): 0.00026655034266817346,
            ('ns_sigma', 'BG'): -0.00035502620690710114, ('ns_sigma', 'v2_t'): -2.7640556065111823e-06,
            ('ns_sigma', 'v2_a'): 4.692003518761862e-08, ('ns_sigma', 'v4_t'): 3.3575404158283206e-07,
            ('ns_sigma', 'v4_a'): 2.3485231088135886e-07, ('ns_sigma', 'v3'): -1.4273558060264512e-06,
            ('ns_sigma', 'B'): -2.600804727681495e-05, ('as_sigma', 'ns_amplitude'): 0.004586628583904665,
            ('as_sigma', 'as_amplitude'): 0.008412717771853899, ('as_sigma', 'ns_sigma'): 0.00026655034266817346,
            ('as_sigma', 'as_sigma'): 0.006070036665554977, ('as_sigma', 'BG'): -0.00206225002748495,
            ('as_sigma', 'v2_t'): -6.500023012349592e-06, ('as_sigma', 'v2_a'): 3.3860989603375093e-06,
            ('as_sigma', 'v4_t'): 2.1962728737091226e-06, ('as_sigma', 'v4_a'): 3.0421272676325655e-06,
            ('as_sigma', 'v3'): -5.4028432040217185e-05, ('as_sigma', 'B'): -0.0017156715899985726,
            ('BG', 'ns_amplitude'): -0.004019509691337873, ('BG', 'as_amplitude'): -0.005534282484339136,
            ('BG', 'ns_sigma'): -0.00035502620690710114, ('BG', 'as_sigma'): -0.00206225002748495,
            ('BG', 'BG'): 0.0024165089641249695, ('BG', 'v2_t'): 1.1909535850558192e-05,
            ('BG', 'v2_a'): -9.235738758378376e-07, ('BG', 'v4_t'): -1.1225041117879457e-06,
            ('BG', 'v4_a'): -4.984057137464932e-07, ('BG', 'v3'): 1.752340820569268e-05,
            ('BG', 'B'): 0.00048338251767429316, ('v2_t', 'ns_amplitude'): -3.5013665434579806e-05,
            ('v2_t', 'as_amplitude'): -3.973113571976044e-05, ('v2_t', 'ns_sigma'): -2.7640556065111823e-06,
            ('v2_t', 'as_sigma'): -6.500023012349592e-06, ('v2_t', 'BG'): 1.1909535850558192e-05,
            ('v2_t', 'v2_t'): 6.320254289545203e-06, ('v2_t', 'v2_a'): 1.0042970964403218e-07,
            ('v2_t', 'v4_t'): -3.820177656965649e-07, ('v2_t', 'v4_a'): 1.175863881401016e-07,
            ('v2_t', 'v3'): 7.136379479651473e-08, ('v2_t', 'B'): -4.4784727962737335e-05,
            ('v2_a', 'ns_amplitude'): 3.6570973251680785e-06, ('v2_a', 'as_amplitude'): 2.2158781868346477e-06,
            ('v2_a', 'ns_sigma'): 4.692003518761862e-08, ('v2_a', 'as_sigma'): 3.3860989603375093e-06,
            ('v2_a', 'BG'): -9.235738758378376e-07, ('v2_a', 'v2_t'): 1.0042970964403218e-07,
            ('v2_a', 'v2_a'): 1.5593189473215385e-05, ('v2_a', 'v4_t'): -5.362285833796496e-08,
            ('v2_a', 'v4_a'): 3.2849588077829256e-07, ('v2_a', 'v3'): -8.720385751786689e-08,
            ('v2_a', 'B'): 2.7828543786649263e-08, ('v4_t', 'ns_amplitude'): 3.047333679513599e-06,
            ('v4_t', 'as_amplitude'): 3.993982077267372e-06, ('v4_t', 'ns_sigma'): 3.3575404158283206e-07,
            ('v4_t', 'as_sigma'): 2.1962728737091226e-06, ('v4_t', 'BG'): -1.1225041117879457e-06,
            ('v4_t', 'v2_t'): -3.820177656965649e-07, ('v4_t', 'v2_a'): -5.362285833796496e-08,
            ('v4_t', 'v4_t'): 1.0416749518759913e-05, ('v4_t', 'v4_a'): -2.2806392429457846e-07,
            ('v4_t', 'v3'): -2.4769876813646587e-08, ('v4_t', 'B'): 2.3164202191932045e-06,
            ('v4_a', 'ns_amplitude'): 1.0803267317539679e-06, ('v4_a', 'as_amplitude'): 2.0448291125666287e-06,
            ('v4_a', 'ns_sigma'): 2.3485231088135886e-07, ('v4_a', 'as_sigma'): 3.0421272676325655e-06,
            ('v4_a', 'BG'): -4.984057137464932e-07, ('v4_a', 'v2_t'): 1.175863881401016e-07,
            ('v4_a', 'v2_a'): 3.2849588077829256e-07, ('v4_a', 'v4_t'): -2.2806392429457846e-07,
            ('v4_a', 'v4_a'): 2.085553317728389e-05, ('v4_a', 'v3'): -3.6541969118161186e-08,
            ('v4_a', 'B'): -3.077207877523124e-06, ('v3', 'ns_amplitude'): -6.630890180366187e-05,
            ('v3', 'as_amplitude'): -4.486572540586001e-05, ('v3', 'ns_sigma'): -1.4273558060264512e-06,
            ('v3', 'as_sigma'): -5.4028432040217185e-05, ('v3', 'BG'): 1.752340820569268e-05,
            ('v3', 'v2_t'): 7.136379479651473e-08, ('v3', 'v2_a'): -8.720385751786689e-08,
            ('v3', 'v4_t'): -2.4769876813646587e-08, ('v3', 'v4_a'): -3.6541969118161186e-08,
            ('v3', 'v3'): 1.3740774655449359e-06, ('v3', 'B'): 4.433272832354065e-05,
            ('B', 'ns_amplitude'): -0.0019040291327373684, ('B', 'as_amplitude'): -0.0011687806471112946,
            ('B', 'ns_sigma'): -2.600804727681495e-05, ('B', 'as_sigma'): -0.0017156715899985726,
            ('B', 'BG'): 0.00048338251767429316, ('B', 'v2_t'): -4.4784727962737335e-05,
            ('B', 'v2_a'): 2.7828543786649263e-08, ('B', 'v4_t'): 2.3164202191932045e-06,
            ('B', 'v4_a'): -3.077207877523124e-06, ('B', 'v3'): 4.433272832354065e-05,
            ('B', 'B'): 0.08004222963133066
        },
        x = np.array([-1.48352986, -1.30899694, -1.13446401, -0.95993109, -0.78539816,
                      -0.61086524, -0.43633231, -0.26179939, -0.08726646,  0.08726646,  # noqa: E241
                       0.26179939,  0.43633231,  0.61086524,  0.78539816,  0.95993109,  # noqa: E241
                       1.13446401,  1.30899694,  1.48352986,  1.65806279,  1.83259571,  # noqa: E241
                       2.00712864,  2.18166156,  2.35619449,  2.53072742,  2.70526034,  # noqa: E241
                       2.87979327,  3.05432619,  3.22885912,  3.40339204,  3.57792497,  # noqa: E241
                       3.75245789,  3.92699082,  4.10152374,  4.27605667,  4.45058959,  # noqa: E241
                       4.62512252]),
        n_fit_data_points = 90,
        minimum_val = 82.97071829310454,
    )

    # Run the fit
    rp_fit, data = example.run_inclusive_signal_fit(input_filename = sample_data_filename)

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result, expected_fit_result = expected_fit_result) is True

@pytest.mark.slow
def test_background_fit(setup_integration_tests):
    """ Integration test for the background fit.

    This uses the sample data in the ``testFiles`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    expected_fit_result = base.RPFitResult(
        parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v1', 'v3'],
        free_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v3'],
        fixed_parameters = ['v1'],
        values_at_minimum = {
            'B': 74.19749807006653,
            'v2_t': 2.1510320818984852e-07, 'v2_a': 0.04187686895800907,
            'v4_t': 0.002154202588079246, 'v4_a': 0.0026027686260190197,
            'v1': 0.0, 'v3': 0.004734080067730018
        },
        covariance_matrix = {
            ('B', 'B'): 0.08644453056066083, ('B', 'v2_t'): -1.4931622683135787e-07,
            ('B', 'v2_a'): -1.2408212424081687e-05, ('B', 'v4_t'): -1.8803445595574103e-06,
            ('B', 'v4_a'): -5.2308644575833155e-06, ('B', 'v3'): 0.00024988796340065603,
            ('v2_t', 'B'): -1.4931622683135787e-07, ('v2_t', 'v2_t'): 1.6962140371680398e-08,
            ('v2_t', 'v2_a'): -1.770520725261621e-10, ('v2_t', 'v4_t'): -1.2586373334238697e-09,
            ('v2_t', 'v4_a'): 3.3582302490263605e-10, ('v2_t', 'v3'): 1.8498515785940135e-10,
            ('v2_a', 'B'): -1.2408212424081687e-05, ('v2_a', 'v2_t'): -1.770520725261621e-10,
            ('v2_a', 'v2_a'): 1.557722396332187e-05, ('v2_a', 'v4_t'): -4.1004968685104585e-08,
            ('v2_a', 'v4_a'): 3.2984592258523553e-07, ('v2_a', 'v3'): -5.027135478191435e-07,
            ('v4_t', 'B'): -1.8803445595574103e-06, ('v4_t', 'v2_t'): -1.2586373334238697e-09,
            ('v4_t', 'v2_a'): -4.1004968685104585e-08, ('v4_t', 'v4_t'): 1.0402807489396649e-05,
            ('v4_t', 'v4_a'): -1.135195994956174e-08, ('v4_t', 'v3'): -6.546979210938196e-08,
            ('v4_a', 'B'): -5.2308644575833155e-06, ('v4_a', 'v2_t'): 3.3582302490263605e-10,
            ('v4_a', 'v2_a'): 3.2984592258523553e-07, ('v4_a', 'v4_t'): -1.135195994956174e-08,
            ('v4_a', 'v4_a'): 2.076138823340561e-05, ('v4_a', 'v3'): -1.281475223558177e-07,
            ('v3', 'B'): 0.00024988796340065603, ('v3', 'v2_t'): 1.8498515785940135e-10,
            ('v3', 'v2_a'): -5.027135478191435e-07, ('v3', 'v4_t'): -6.546979210938196e-08,
            ('v3', 'v4_a'): -1.281475223558177e-07, ('v3', 'v3'): 7.830350113052414e-06
        },
        x = np.array([-1.48352986, -1.30899694, -1.13446401, -0.95993109, -0.78539816,  # noqa: E241
                      -0.61086524, -0.43633231, -0.26179939, -0.08726646,  0.08726646,  # noqa: E241
                       0.26179939,  0.43633231,  0.61086524,  0.78539816,  0.95993109,  # noqa: E241
                       1.13446401,  1.30899694,  1.48352986,  1.65806279,  1.83259571,  # noqa: E241
                       2.00712864,  2.18166156,  2.35619449,  2.53072742,  2.70526034,  # noqa: E241
                       2.87979327,  3.05432619,  3.22885912,  3.40339204,  3.57792497,  # noqa: E241
                       3.75245789,  3.92699082,  4.10152374,  4.27605667,  4.45058959,  # noqa: E241
                       4.62512252]),
        n_fit_data_points = 54,
        minimum_val = 38.308756460200534,
    )

    # Run the fit
    rp_fit, data = example.run_background_fit(input_filename = sample_data_filename)

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result, expected_fit_result = expected_fit_result) is True

