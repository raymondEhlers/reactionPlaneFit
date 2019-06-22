#!/usr/bin/env python

""" Integration tests for the inclusive signal and background only fits.

The tests are performed by leveraging the example fit code, as it runs an entire fit.

By using integration tests, we can tests large positions of the package more rapidly.
This of course trades granularity and may allow some branches to slip through, but the
time savings is worth it.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import pkg_resources
import probfit
import pytest
import tempfile
from typing import Any, Optional, Tuple, TYPE_CHECKING

from reaction_plane_fit import base
from reaction_plane_fit import example
from reaction_plane_fit import functions
from reaction_plane_fit import plot
from reaction_plane_fit import three_orientations

if TYPE_CHECKING:
    import iminuit  # noqa: F401

# Typing helpers
# This is a tuple of the matplotlib figure and axes. However, we only specify Any here
# because we don't want an explicit package dependency on matplotlib.
Axes = Any
Figure = Any
DrawResult = Tuple[Figure, Axes]

logger = logging.getLogger(__name__)

@pytest.fixture  # type: ignore
def setup_integration_tests(logging_mixin: Any) -> str:
    """ Setup shared expected values for the fit integration tests. """
    sample_data_filename = pkg_resources.resource_filename("reaction_plane_fit.sample_data", "three_orientations.root")

    # Quiet down matplotlib if it's available. It's extremely noisy!
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    return sample_data_filename

def compare_fit_result_to_expected(fit_result: base.FitResult, expected_fit_result: base.FitResult, minuit: Optional["iminuit.Minuit"] = None) -> bool:
    """ Helper function to compare a fit result to an expected fit result.

    Args:
        fit_result: The calculated fit result.
        expected_fit_result: The expected fit result.
        minuit: Minuit from the fit.
    Returns:
        bool: True if the fit results are the same.
    """
    logger.debug(f"fit_result.parameters: {fit_result.parameters}")
    assert fit_result.parameters == expected_fit_result.parameters
    assert fit_result.free_parameters == expected_fit_result.free_parameters
    assert fit_result.fixed_parameters == expected_fit_result.fixed_parameters
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.values_at_minimum.keys()) == list(expected_fit_result.values_at_minimum.keys())
    # Need the extra tolerance to work on other systems.
    np.testing.assert_allclose(
        list(fit_result.values_at_minimum.values()),
        list(expected_fit_result.values_at_minimum.values()),
        atol = 1e-2, rtol = 0
    )
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.errors_on_parameters.keys()) == list(expected_fit_result.errors_on_parameters.keys())
    # Need the extra tolerance to work on other systems.
    np.testing.assert_allclose(
        list(fit_result.errors_on_parameters.values()),
        list(expected_fit_result.errors_on_parameters.values()),
        atol = 1e-3, rtol = 0
    )
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.covariance_matrix.keys()) == list(expected_fit_result.covariance_matrix.keys())
    # Need the extra tolerance to work on other systems.
    np.testing.assert_allclose(
        list(fit_result.covariance_matrix.values()),
        list(expected_fit_result.covariance_matrix.values()),
        atol = 1e-3, rtol = 0
    )
    # Compare correlation matrices to minuit
    if minuit:
        correlation = minuit.np_matrix(correlation = True).reshape(-1)
        np.testing.assert_allclose(
            correlation,
            list(expected_fit_result.correlation_matrix.values()),
            atol = 0.1, rtol = 0,
        )
    # Check that the correlation matrix diagonal is one as a sanity check
    correlation_diagonal = [expected_fit_result.correlation_matrix[(n, n)] for n in expected_fit_result.free_parameters]
    assert np.allclose(correlation_diagonal, np.ones(len(correlation_diagonal)))

    # Check the general fit description
    np.testing.assert_allclose(fit_result.x, expected_fit_result.x, atol = 1e-5, rtol = 0)
    assert fit_result.n_fit_data_points == expected_fit_result.n_fit_data_points
    assert np.isclose(fit_result.minimum_val, expected_fit_result.minimum_val, atol = 1e-2)

    # Calculated values
    assert fit_result.nDOF == expected_fit_result.nDOF

    # Check the errors
    np.testing.assert_allclose(fit_result.errors, expected_fit_result.errors, atol = 2e-2, rtol = 0)

    # If all assertions passed, then return True to indicate the success.
    return True

@pytest.mark.slow  # type: ignore
@pytest.mark.mpl_image_compare(tolerance = 5)  # type: ignore
def test_inclusive_signal_fit(setup_integration_tests: Any) -> Figure:
    """ Integration test for the inclusive signal fit.

    This uses the sample data in the ``testFiles`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    # NOTE: They are not calculated independently, so there are most like regression tests.
    expected_fit_result = base.FitResult(
        parameters = [
            "ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "signal_pedestal", "BG",
            "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3", "B",
        ],
        free_parameters = [
            "ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "BG",
            "v2_t", "v2_a", "v4_t", "v4_a", "v3", "B",
        ],
        fixed_parameters = ["signal_pedestal", "v1"],
        values_at_minimum = {
            'ns_amplitude': 1.7933561625700922, 'as_amplitude': 1.0001072229610664,
            'ns_sigma': 0.2531797052038343, 'as_sigma': 0.4304508803936734,
            'signal_pedestal': 0.0, 'BG': 24.51231187224201,
            'v2_t': 1.635883107545255e-05, 'v2_a': 0.04207445772474305,
            'v4_t': 0.0021469295947451894, 'v4_a': 0.002584005026287889,
            'v1': 0.0, 'v3': 0.0016802003999212695,
            'B': 74.09951310294166,
        },
        errors_on_parameters = {
            'ns_amplitude': 0.12839073902939924, 'as_amplitude': 0.1601590707638878,
            'ns_sigma': 0.01756754811099917, 'as_sigma': 0.0771999164701046,
            'signal_pedestal': 1.0, 'BG': 0.04915769800151182,
            'v2_t': 0.09027663461261039, 'v2_a': 0.0039482883973311345,
            'v4_t': 0.003222254298287666, 'v4_a': 0.004554445381827463,
            'v1': 1.0, 'v3': 0.001172204666172523,
            'B': 0.28291657094270306,
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
        errors = [],
    )
    expected_signal_parameters = [
        'ns_amplitude', 'as_amplitude', 'ns_sigma', 'as_sigma', 'signal_pedestal',
        'BG', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v1', 'v3'
    ]
    expected_signal_free_parameters = [
        'ns_amplitude', 'as_amplitude', 'ns_sigma', 'as_sigma',
        'BG', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v3'
    ]
    expected_background_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v1', 'v3']
    expected_background_free_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v3']
    x_component = np.array(expected_fit_result.x[:int(len(expected_fit_result.x) / 2)])
    expected_components = {
        base.FitType(region='signal', orientation='inclusive'): base.FitResult(
            parameters = expected_signal_parameters,
            free_parameters = expected_signal_free_parameters,
            fixed_parameters = ['signal_pedestal', 'v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_signal_parameters},
            errors_on_parameters = {k: expected_fit_result.errors_on_parameters[k] for k in expected_signal_parameters},
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_signal_free_parameters for k2 in expected_signal_free_parameters},
            x = expected_fit_result.x,
            n_fit_data_points = 36,
            minimum_val = 43.47809398077163,
            errors = np.array([
                0.04659353, 0.05320476, 0.06173382, 0.06129561, 0.05044053,
                0.05628865, 0.10358349, 0.09225062, 0.13088703, 0.13088703,
                0.09225063, 0.10358349, 0.05628864, 0.05044054, 0.06129673,
                0.06173997, 0.05318416, 0.04603239, 0.05215069, 0.0630976 ,  # noqa: E203
                0.06490267, 0.0595903 , 0.06622332, 0.07684584, 0.06925706,  # noqa: E203
                0.07379214, 0.10335831, 0.10335831, 0.07379214, 0.06925706,
                0.07684584, 0.06622332, 0.0595903 , 0.06490267, 0.0630976 ,  # noqa: E203
                0.05215069]),
        ),
        base.FitType(region='background', orientation='in_plane'): base.FitResult(
            parameters = expected_background_parameters,
            free_parameters=expected_background_free_parameters,
            fixed_parameters=['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            x = x_component,
            n_fit_data_points = 18,
            minimum_val = 17.470196626377145,
            errors = np.array([
                0.22600626, 0.20236964, 0.1792622 , 0.17417331, 0.17607922,  # noqa: E203
                0.17589761, 0.1895709 , 0.22364286, 0.25277766, 0.25277766,  # noqa: E203
                0.22364286, 0.1895709 , 0.17589761, 0.17607922, 0.17417331,  # noqa: E203
                0.1792622 , 0.20236964, 0.22600626, 0.2282912 , 0.20927409,  # noqa: E203
                0.18976985, 0.18483703, 0.18363486, 0.17856864, 0.18720743,
                0.21845047, 0.24672188, 0.24672188, 0.21845047, 0.18720743,
                0.17856864, 0.18363486, 0.18483703, 0.18976985, 0.20927409,
                0.2282912])
        ),
        base.FitType(region='background', orientation='mid_plane'): base.FitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            x = x_component,
            n_fit_data_points = 18,
            minimum_val = 12.466924663501008,
            errors = np.array([
                0.235723  , 0.18639393, 0.16976855, 0.22061279, 0.24931356,  # noqa: E203
                0.21663034, 0.16738339, 0.19307235, 0.24641435, 0.24641435,
                0.19307235, 0.16738339, 0.21663034, 0.24931356, 0.22061279,
                0.16976855, 0.18639393, 0.235723  , 0.23774675, 0.19306969,  # noqa: E203
                0.1791784 , 0.22759284, 0.25377834, 0.21855863, 0.16472916,  # noqa: E203
                0.18638008, 0.23902805, 0.23902805, 0.18638008, 0.16472916,
                0.21855863, 0.25377834, 0.22759284, 0.1791784 , 0.19306969,  # noqa: E203
                0.23774675])
        ),
        base.FitType(region='background', orientation='out_of_plane'): base.FitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            x = x_component,
            n_fit_data_points = 18,
            minimum_val = 9.559659824619239,
            errors = np.array([
                0.24979395, 0.2269838 , 0.20247939, 0.19254136, 0.18836174,  # noqa: E203
                0.18163514, 0.1881116 , 0.21693021, 0.24380993, 0.24380993,  # noqa: E203
                0.21693021, 0.1881116 , 0.18163514, 0.18836174, 0.19254136,  # noqa: E203
                0.20247939, 0.2269838 , 0.24979395, 0.25079664, 0.23018009,  # noqa: E203
                0.20785398, 0.19878404, 0.19344469, 0.18368386, 0.18606088,
                0.2120099 , 0.23782647, 0.23782647, 0.2120099 , 0.18606088,  # noqa: E203
                0.18368386, 0.19344469, 0.19878404, 0.20785398, 0.23018009,
                0.25079664])
        ),
    }

    # Run the fit
    # NOTE: The user_arguments are the same as the defaults to ensure that they don't change the fit, but we specify
    #       one to test the logic of how they are set.
    rp_fit, data, minuit = example.run_inclusive_signal_fit(
        input_filename = sample_data_filename,
        user_arguments = {"v2_t": 0.02},
    )

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result,
                                          expected_fit_result = expected_fit_result,
                                          minuit = minuit) is True
    # Check the components
    for fit_type, fit_component in rp_fit.components.items():
        assert compare_fit_result_to_expected(fit_result = fit_component.fit_result,
                                              expected_fit_result = expected_components[fit_type]) is True

    # Draw and check the resulting image. It is checked by returning the figure.
    # We skip the residual plot because it is hard to compare multiple images from in the same test.
    # Instead, we get around this by comparing the residual in test_background_fit(...)
    fig, ax = plot.draw_fit(rp_fit = rp_fit, data = data, filename = "")
    return fig

@pytest.mark.slow  # type: ignore
@pytest.mark.mpl_image_compare(tolerance = 5)  # type: ignore
def test_background_fit(setup_integration_tests: Any) -> Figure:
    """ Integration test for the background fit.

    This uses the sample data in the ``testFiles`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    expected_fit_result = base.FitResult(
        parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v1', 'v3'],
        free_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v3'],
        fixed_parameters = ['v1'],
        values_at_minimum = {
            'B': 74.19749807006653,
            'v2_t': 2.1510320818984852e-07, 'v2_a': 0.04187686895800907,
            'v4_t': 0.002154202588079246, 'v4_a': 0.0026027686260190197,
            'v1': 0.0, 'v3': 0.004734080067730018
        },
        errors_on_parameters = {
            'B': 0.2940144452610056,
            'v2_t': 0.01945621214508514, 'v2_a': 0.003946265121221856,
            'v4_t': 0.003220126599587231, 'v4_a': 0.004544297895328175,
            'v1': 1.0, 'v3': 0.0027982058671116375
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
        errors = [],
    )
    expected_background_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v1', 'v3']
    expected_background_free_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v3']
    x_component = np.array(expected_fit_result.x[:int(len(expected_fit_result.x) / 2)])
    expected_components = {
        base.FitType(region='background', orientation='in_plane'): base.FitResult(
            parameters = expected_background_parameters,
            free_parameters=expected_background_free_parameters,
            fixed_parameters=['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            x = x_component,
            n_fit_data_points = expected_fit_result.n_fit_data_points / 3,
            minimum_val = 16.73538215589725,
            errors = np.array([
                0.2093359 , 0.19255153, 0.18174519, 0.17607629, 0.16211076,  # noqa: E203
                0.14768848, 0.17124385, 0.22832949, 0.27169606, 0.27169606,
                0.22832949, 0.17124385, 0.14768848, 0.16211076, 0.17607629,
                0.18174519, 0.19255153, 0.2093359 , 0.22090577, 0.22490269,  # noqa: E203
                0.22619674, 0.22037282, 0.19623391, 0.16124127, 0.15950035,
                0.20546865, 0.2466069 , 0.2466069 , 0.20546865, 0.15950035,  # noqa: E203
                0.16124127, 0.19623391, 0.22037282, 0.22619674, 0.22490269,
                0.22090577])
        ),
        base.FitType(region='background', orientation='mid_plane'): base.FitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            x = x_component,
            n_fit_data_points = expected_fit_result.n_fit_data_points / 3,
            minimum_val = 11.457530895031944,
            errors = np.array([
                0.23699861, 0.19647139, 0.19316462, 0.23845882, 0.25528958,
                0.21538816, 0.17786381, 0.22716874, 0.29022977, 0.29022977,
                0.22716874, 0.17786381, 0.21538816, 0.25528958, 0.23845882,
                0.19316462, 0.19647139, 0.23699861, 0.24705662, 0.22716873,
                0.23335852, 0.27103529, 0.27778609, 0.22547584, 0.16444424,
                0.19647136, 0.25702783, 0.25702783, 0.19647136, 0.16444424,
                0.22547584, 0.27778609, 0.27103529, 0.23335852, 0.22716873,
                0.24705662])
        ),
        base.FitType(region='background', orientation='out_of_plane'): base.FitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            x = x_component,
            n_fit_data_points = expected_fit_result.n_fit_data_points / 3,
            minimum_val = 10.115864298085231,
            errors = np.array([
                0.22106564, 0.20550612, 0.19201483, 0.18118305, 0.16215477,
                0.14395781, 0.16678903, 0.22492152, 0.2689348 , 0.2689348 ,  # noqa: E203
                0.22492152, 0.16678903, 0.14395781, 0.16215477, 0.18118305,
                0.19201483, 0.20550612, 0.22106564, 0.22880924, 0.22835903,
                0.22663356, 0.22011445, 0.19626572, 0.15946394, 0.15152662,
                0.19257799, 0.23184201, 0.23184201, 0.19257799, 0.15152662,
                0.15946394, 0.19626572, 0.22011445, 0.22663356, 0.22835903,
                0.22880924])
        ),
    }

    # Run the fit
    # NOTE: The user_arguments are the same as the defaults to ensure that they don't change the fit, but we specify
    #       one to test the logic of how they are set.
    rp_fit, data, minuit = example.run_background_fit(
        input_filename = sample_data_filename,
        user_arguments = {"v2_t": 0.02},
    )

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result,
                                          expected_fit_result = expected_fit_result,
                                          minuit = minuit) is True
    # Check the components
    for fit_type, fit_component in rp_fit.components.items():
        assert compare_fit_result_to_expected(fit_result = fit_component.fit_result,
                                              expected_fit_result = expected_components[fit_type]) is True

    # Check standard summary components
    # Crreate the full set of components and check them out.
    summary_components = rp_fit.create_full_set_of_components()

    # Make sure that we have all of the components (including the inclusive)
    assert len(summary_components) == len(rp_fit._rp_orientations)

    # Check the inclusive summary component.
    inclusive_component = summary_components["inclusive"]
    assert inclusive_component.fit_function == three_orientations.constrained_inclusive_background
    assert inclusive_component.fit_result.parameters == expected_background_parameters
    # Check values
    values_at_minimum = expected_fit_result.values_at_minimum
    assert list(inclusive_component.fit_result.values_at_minimum) == list(values_at_minimum)
    np.testing.assert_allclose(
        list(inclusive_component.fit_result.values_at_minimum.values()), list(values_at_minimum.values()),
        atol = 1e-5, rtol = 0,
    )
    # We want to compare against the fourier values.
    values_at_minimum["B"] = values_at_minimum["B"] / 3
    x = expected_fit_result.x
    expected_inclusive_component_values = probfit.nputil.vector_apply(functions.fourier, x, *list(values_at_minimum.values()))
    np.testing.assert_allclose(inclusive_component.evaluate_fit(x), expected_inclusive_component_values)

    # Check that we've properly transfered the fit results to the summary components.
    for fit_type, fit_component in rp_fit.components.items():
        if fit_type.orientation in summary_components:
            assert compare_fit_result_to_expected(fit_result = summary_components[fit_type.orientation].fit_result,
                                                  expected_fit_result = fit_component.fit_result) is True

    # Draw and check the resulting image. It is checked by returning the figure.
    # We skip the fit plot because it is hard to compare multiple images from in the same test.
    # Instead, we get around this by comparing the fit in test_inclusive_signal_fit(...)
    fig, ax = plot.draw_residual(rp_fit = rp_fit, data = data, filename = "")
    return fig

@pytest.mark.slow  # type: ignore
@pytest.mark.parametrize("calculate_correlation_matrix_before_writing", [  # type: ignore
    False, True,
], ids = ["Don't calculate correlation matrix", "Calculate correlation matrix"])
@pytest.mark.parametrize("example_module_func, fit_object", [  # type: ignore
    (example.run_background_fit, three_orientations.BackgroundFit),
    (example.run_inclusive_signal_fit, three_orientations.InclusiveSignalFit),
], ids = ["Background", "Inclusive signal"])
def test_write_and_read_result_in_class(logging_mixin: Any, setup_integration_tests: Any, calculate_correlation_matrix_before_writing: bool,
                                        example_module_func: Any, fit_object: Any) -> None:
    """ Test storing results via the pachyderm `yaml` module.

    Note:
        This ends up as an integration test because we actually use the yaml module
        to read and write to a file instead of mocking it.
    """
    # Setup
    sample_data_filename = setup_integration_tests

    expected_rp_fit, data, minuit = example_module_func(
        input_filename = sample_data_filename,
        user_arguments = {"v2_t": 0.02},
    )

    # Same definition as used in the example fit.
    # We must be certain _not_ to run the fit for this object because we are going to use it for comparison.
    rp_fit = fit_object(
        resolution_parameters = {"R22": 1, "R42": 1, "R62": 1, "R82": 1},
        use_log_likelihood = False,
        signal_region = (0, 0.6),
        background_region = (0.8, 1.2),
    )

    if calculate_correlation_matrix_before_writing:
        # Don't need to actually do anything with it - just calculate it.
        expected_rp_fit.fit_result.correlation_matrix

    # Write to and then read from a file.
    with tempfile.NamedTemporaryFile(mode = "r+") as f:
        # Use the performed fit to write the fit result.
        expected_rp_fit.write_fit_results(filename = f.name)
        # Move back to beginning
        f.seek(0)
        # Load the fit into the newly created fit object (which hasn't run the fit).
        rp_fit.read_fit_results(filename = f.name)

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result,
                                          expected_fit_result = expected_rp_fit.fit_result,
                                          minuit = minuit) is True
    # Check the correlation matrix are the same in the fit objects (since we are testing if the object
    # is preserved during the YAML round trip)
    assert list(rp_fit.fit_result.correlation_matrix.keys()) == list(expected_rp_fit.fit_result.correlation_matrix.keys())
    np.testing.assert_allclose(list(rp_fit.fit_result.correlation_matrix.values()),
                               list(expected_rp_fit.fit_result.correlation_matrix.values()))
    # Check the components
    for (fit_type, fit_component), (expected_fit_type, expected_fit_component) in \
            zip(rp_fit.components.items(), expected_rp_fit.components.items()):
        assert fit_type == expected_fit_type
        assert compare_fit_result_to_expected(fit_result = fit_component.fit_result,
                                              expected_fit_result = expected_fit_component.fit_result) is True
        # Check the correlation matrix are the same in the fit objects (since we are testing if the object
        # is preserved during the YAML round trip)
        assert list(fit_component.fit_result.correlation_matrix.keys()) == \
            list(expected_fit_component.fit_result.correlation_matrix.keys())
        np.testing.assert_allclose(list(fit_component.fit_result.correlation_matrix.values()),
                                   list(expected_fit_component.fit_result.correlation_matrix.values()))

def test_invalid_arguments(logging_mixin: Any, setup_integration_tests: Any) -> None:
    """ Test detection for invalid arguments.

    This doesn't really need to be integration test, but it's very convenient to use
    the integration test setup and the example module, so we perform the test here.
    """
    # Setup
    sample_data_filename = setup_integration_tests

    # Pass an invalid argument.
    with pytest.raises(ValueError) as exception_info:
        example.run_background_fit(
            input_filename = sample_data_filename,
            user_arguments = {"abcdefg": 0.02},
        )

    # Check that the invalid argument was caught successfully.
    assert "User argument abcdefg" in exception_info.value.args[0]

@pytest.mark.slow  # type: ignore
@pytest.mark.mpl_image_compare(tolerance = 5)  # type: ignore
def test_signal_fit(setup_integration_tests: Any) -> Figure:
    """ Integration test for the (differential) signal fit.

    This uses the sample data in the ``testFiles`` directory.
    """

