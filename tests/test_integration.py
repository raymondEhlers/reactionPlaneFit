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
    expected_fit_result = base.RPFitResult(
        parameters = ["ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "signal_pedestal", "BG", "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3", "B"],
        free_parameters = ["ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "BG", "v2_t", "v2_a", "v4_t", "v4_a", "v3", "B"],
        fixed_parameters = ["signal_pedestal", "v1"],
        minimum_val = 203.69271481570132,
        values_at_minimum = {
            "ns_amplitude": 9.535191718061213,
            "as_amplitude": 0.715433199661658,
            "ns_sigma": 0.031084277626599027,
            "as_sigma": 0.30018344238543426,
            "signal_pedestal": 0.0,
            "BG": 24.70454068642269,
            "v2_t": 0.0023768855820206214,
            "v2_a": 0.04213395205728768,
            "v4_t": 0.0020326769397844002,
            "v4_a": 0.0026281123008189133,
            "v1": 0.0,
            "v3": 0.004964050654580873,
            "B": 74.18950515937316
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
        covariance_matrix = {
            ('ns_amplitude', 'ns_amplitude'): 65.97848122539429, ('ns_amplitude', 'as_amplitude'): 0.0010882246155804035, ('ns_amplitude', 'ns_sigma'): -0.031174652229125174,
            ('ns_amplitude', 'as_sigma'): 0.0011391352596629488, ('ns_amplitude', 'BG'): -0.0021769268315612947, ('ns_amplitude', 'v2_t'): -3.643615409647639e-05,
            ('ns_amplitude', 'v2_a'): -2.7486345210443727e-06, ('ns_amplitude', 'v4_t'): 2.6221264192458606e-08, ('ns_amplitude', 'v4_a'): -3.2620638671688712e-06,
            ('ns_amplitude', 'v3'): -7.509073282301293e-05, ('ns_amplitude', 'B'): -0.0021340800723423418, ('as_amplitude', 'ns_amplitude'): 0.0010882246155804035,
            ('as_amplitude', 'as_amplitude'): 0.011649165220689038, ('as_amplitude', 'ns_sigma'): 6.038533176985434e-07, ('as_amplitude', 'as_sigma'): 0.0021141779249445234,
            ('as_amplitude', 'BG'): -0.0018626553039241148, ('as_amplitude', 'v2_t'): -2.4099172125623146e-05, ('as_amplitude', 'v2_a'): -7.400290440982007e-06,
            ('as_amplitude', 'v4_t'): 4.7233277550310295e-07, ('as_amplitude', 'v4_a'): -1.727563190986312e-06, ('as_amplitude', 'v3'): 2.8089458785232937e-05,
            ('as_amplitude', 'B'): 0.0010563412472705208, ('ns_sigma', 'ns_amplitude'): -0.031174652229125174, ('ns_sigma', 'as_amplitude'): 6.038533176985434e-07,
            ('ns_sigma', 'ns_sigma'): 1.4807279735680899e-05, ('ns_sigma', 'as_sigma'): 6.321018026826155e-07, ('ns_sigma', 'BG'): -1.207969101048698e-06,
            ('ns_sigma', 'v2_t'): -2.021815303846541e-08, ('ns_sigma', 'v2_a'): -1.5249504783137407e-09, ('ns_sigma', 'v4_t'): 1.4673309841939518e-11,
            ('ns_sigma', 'v4_a'): -1.8098339787270487e-09, ('ns_sigma', 'v3'): -4.166752733012761e-08, ('ns_sigma', 'B'): -1.1841926732206695e-06,
            ('as_sigma', 'ns_amplitude'): 0.0011391352596629488, ('as_sigma', 'as_amplitude'): 0.0021141779249445234, ('as_sigma', 'ns_sigma'): 6.321018026826155e-07,
            ('as_sigma', 'as_sigma'): 0.0016088145639437173, ('as_sigma', 'BG'): -0.0003710169187360313, ('as_sigma', 'v2_t'): -2.5021143596574075e-06,
            ('as_sigma', 'v2_a'): -1.8857398175509053e-07, ('as_sigma', 'v4_t'): 5.099111774414118e-07, ('as_sigma', 'v4_a'): 5.613386400696906e-07,
            ('as_sigma', 'v3'): -5.001764703913379e-06, ('as_sigma', 'B'): -0.00014167041384876013, ('BG', 'ns_amplitude'): -0.0021769268315612947,
            ('BG', 'as_amplitude'): -0.0018626553039241148, ('BG', 'ns_sigma'): -1.207969101048698e-06, ('BG', 'as_sigma'): -0.0003710169187360313,
            ('BG', 'BG'): 0.001266352538497351, ('BG', 'v2_t'): 5.15391056401692e-06, ('BG', 'v2_a'): 1.3131832049898974e-06,
            ('BG', 'v4_t'): -8.834892318083977e-08, ('BG', 'v4_a'): 3.758447352931115e-07, ('BG', 'v3'): -2.2569529485033143e-06,
            ('BG', 'B'): -0.00010688089558544799, ('v2_t', 'ns_amplitude'): -3.643615409647639e-05, ('v2_t', 'as_amplitude'): -2.4099172125623146e-05,
            ('v2_t', 'ns_sigma'): -2.021815303846541e-08, ('v2_t', 'as_sigma'): -2.5021143596574075e-06, ('v2_t', 'BG'): 5.15391056401692e-06,
            ('v2_t', 'v2_t'): 7.832854639597043e-06, ('v2_t', 'v2_a'): 8.791104688618191e-07, ('v2_t', 'v4_t'): -3.959950686493469e-07,
            ('v2_t', 'v4_a'): 1.3209772745336868e-07, ('v2_t', 'v3'): -3.3127877617970245e-08, ('v2_t', 'B'): -5.476870802299606e-05,
            ('v2_a', 'ns_amplitude'): -2.7486345210443727e-06, ('v2_a', 'as_amplitude'): -7.400290440982007e-06, ('v2_a', 'ns_sigma'): -1.5249504783137407e-09,
            ('v2_a', 'as_sigma'): -1.8857398175509053e-07, ('v2_a', 'BG'): 1.3131832049898974e-06, ('v2_a', 'v2_t'): 8.791104688618191e-07,
            ('v2_a', 'v2_a'): 1.5588656530458285e-05, ('v2_a', 'v4_t'): -9.438354384801419e-08, ('v2_a', 'v4_a'): 3.3713837455392926e-07,
            ('v2_a', 'v3'): -7.257127814423073e-08, ('v2_a', 'B'): -5.3602389500090455e-06, ('v4_t', 'ns_amplitude'): 2.6221264192458606e-08,
            ('v4_t', 'as_amplitude'): 4.7233277550310295e-07, ('v4_t', 'ns_sigma'): 1.4673309841939518e-11, ('v4_t', 'as_sigma'): 5.099111774414118e-07,
            ('v4_t', 'BG'): -8.834892318083977e-08, ('v4_t', 'v2_t'): -3.959950686493469e-07, ('v4_t', 'v2_a'): -9.438354384801419e-08,
            ('v4_t', 'v4_t'): 1.0385707729044646e-05, ('v4_t', 'v4_a'): -2.0562232143790272e-08, ('v4_t', 'v3'): -7.741987373874017e-09,
            ('v4_t', 'B'): 2.6508215053534483e-06, ('v4_a', 'ns_amplitude'): -3.2620638671688712e-06, ('v4_a', 'as_amplitude'): -1.727563190986312e-06,
            ('v4_a', 'ns_sigma'): -1.8098339787270487e-09, ('v4_a', 'as_sigma'): 5.613386400696906e-07, ('v4_a', 'BG'): 3.758447352931115e-07,
            ('v4_a', 'v2_t'): 1.3209772745336868e-07, ('v4_a', 'v2_a'): 3.3713837455392926e-07, ('v4_a', 'v4_t'): -2.0562232143790272e-08,
            ('v4_a', 'v4_a'): 2.0721388317221166e-05, ('v4_a', 'v3'): -1.8399009117748397e-08, ('v4_a', 'B'): -2.6264587601376175e-06,
            ('v3', 'ns_amplitude'): -7.509073282301293e-05, ('v3', 'as_amplitude'): 2.8089458785232937e-05, ('v3', 'ns_sigma'): -4.166752733012761e-08,
            ('v3', 'as_sigma'): -5.001764703913379e-06, ('v3', 'BG'): -2.2569529485033143e-06, ('v3', 'v2_t'): -3.3127877617970245e-08,
            ('v3', 'v2_a'): -7.257127814423073e-08, ('v3', 'v4_t'): -7.741987373874017e-09, ('v3', 'v4_a'): -1.8399009117748397e-08,
            ('v3', 'v3'): 9.035608650281987e-07, ('v3', 'B'): 2.8905246434617242e-05, ('B', 'ns_amplitude'): -0.0021340800723423418,
            ('B', 'as_amplitude'): 0.0010563412472705208, ('B', 'ns_sigma'): -1.1841926732206695e-06, ('B', 'as_sigma'): -0.00014167041384876013,
            ('B', 'BG'): -0.00010688089558544799, ('B', 'v2_t'): -5.476870802299606e-05, ('B', 'v2_a'): -5.3602389500090455e-06,
            ('B', 'v4_t'): 2.6508215053534483e-06, ('B', 'v4_a'): -2.6264587601376175e-06, ('B', 'v3'): 2.8905246434617242e-05,
            ('B', 'B'): 0.07945823267656031
        }
    )

    # Run the fit
    rp_fit = example.run_inclusive_signal_fit(input_filename = sample_data_filename)

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
        minimum_val = 38.30885185866883,
        values_at_minimum = {
            'B': 74.19719198362343,
            'v2_t': 4.1047425200502197e-07,
            'v2_a': 0.041902582551187284,
            'v4_t': 0.0021751980222733946,
            'v4_a': 0.002604902410456439,
            'v1': 0.0,
            'v3': 0.004738737386540792
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
        covariance_matrix = {
            ('B', 'B'): 0.0864444205776576, ('B', 'v2_t'): -2.584151359928967e-07, ('B', 'v2_a'): -1.2463464910789412e-05,
            ('B', 'v4_t'): -1.909741451585364e-06, ('B', 'v4_a'): -5.249406892684665e-06, ('B', 'v3'): 0.00024986684728795893,
            ('v2_t', 'B'): -2.584151359928967e-07, ('v2_t', 'v2_t'): 3.2183082995480663e-08, ('v2_t', 'v2_a'): -3.06804831133032e-10,
            ('v2_t', 'v4_t'): -2.167091574961501e-09, ('v2_t', 'v4_a'): 5.809022917992412e-10, ('v2_t', 'v3'): 3.1823956802027315e-10,
            ('v2_a', 'B'): -1.2463464910789412e-05, ('v2_a', 'v2_t'): -3.06804831133032e-10, ('v2_a', 'v2_a'): 1.5572305665104053e-05,
            ('v2_a', 'v4_t'): -4.132612827452033e-08, ('v2_a', 'v4_a'): 3.2969323813116065e-07, ('v2_a', 'v3'): -5.027416658704996e-07,
            ('v4_t', 'B'): -1.909741451585364e-06, ('v4_t', 'v2_t'): -2.167091574961501e-09, ('v4_t', 'v2_a'): -4.132612827452033e-08,
            ('v4_t', 'v4_t'): 1.0352910127034277e-05, ('v4_t', 'v4_a'): -1.1407221969160633e-08, ('v4_t', 'v3'): -6.517934630197917e-08,
            ('v4_a', 'B'): -5.249406892684665e-06, ('v4_a', 'v2_t'): 5.809022917992412e-10, ('v4_a', 'v2_a'): 3.2969323813116065e-07,
            ('v4_a', 'v4_t'): -1.1407221969160633e-08, ('v4_a', 'v4_a'): 2.075536703067256e-05, ('v4_a', 'v3'): -1.285496471307051e-07,
            ('v3', 'B'): 0.00024986684728795893, ('v3', 'v2_t'): 3.1823956802027315e-10, ('v3', 'v2_a'): -5.027416658704996e-07,
            ('v3', 'v4_t'): -6.517934630197917e-08, ('v3', 'v4_a'): -1.285496471307051e-07, ('v3', 'v3'): 7.830124349025754e-06
        }
    )

    # Run the fit
    rp_fit = example.run_background_fit(input_filename = sample_data_filename)

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result, expected_fit_result = expected_fit_result) is True

