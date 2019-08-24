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
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING

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

def compare_fit_result_to_expected(fit_result: Union[base.FitResult, base.BaseFitResult],
                                   expected_fit_result: Union[base.BaseFitResult, base.FitResult],
                                   minuit: Optional["iminuit.Minuit"] = None) -> bool:
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

    # Check the errors
    np.testing.assert_allclose(fit_result.errors, expected_fit_result.errors, atol = 2e-2, rtol = 0)

    # Check the general fit description
    if isinstance(fit_result, base.FitResult):
        # Help out mypy...
        assert isinstance(expected_fit_result, base.FitResult)
        # Check the additional attributes
        np.testing.assert_allclose(fit_result.x, expected_fit_result.x, atol = 1e-5, rtol = 0)
        assert fit_result.n_fit_data_points == expected_fit_result.n_fit_data_points
        assert np.isclose(fit_result.minimum_val, expected_fit_result.minimum_val, atol = 1e-2)

        # Calculated values
        assert fit_result.nDOF == expected_fit_result.nDOF

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
        parameters=[
            "ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "signal_pedestal", "BG",
            "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3", "B",
        ],
        free_parameters=[
            "ns_amplitude", "as_amplitude", "ns_sigma", "as_sigma", "BG",
            "v2_t", "v2_a", "v4_t", "v4_a", "v3", "B",
        ],
        fixed_parameters=["signal_pedestal", "v1"],
        values_at_minimum={
            "ns_amplitude": 3.2809411988915516,
            "as_amplitude": 1.8489432297751907,
            "ns_sigma": 0.22868472738561316,
            "as_sigma": 0.4441709704395406,
            "signal_pedestal": 0.0,
            "BG": 58.76378642598284,
            "v2_t": 0.0010001897811286388,
            "v2_a": 0.046815379069918756,
            "v4_t": 4.7683700662992656e-07,
            "v4_a": 0.00590849609503144,
            "v1": 0.0,
            "v3": 0.002632852199485086,
            "B": 175.99722684284168,
        },
        errors_on_parameters={
            "ns_amplitude": 0.22013880427482202,
            "as_amplitude": 0.28265751518882776,
            "ns_sigma": 0.0158226305008205,
            "as_sigma": 0.07218259727767867,
            "signal_pedestal": 1.0,
            "BG": 0.086624204119925,
            "v2_t": 0.0002163040833313033,
            "v2_a": 0.0027335037481057353,
            "v4_t": 0.0008960281450342555,
            "v4_a": 0.0031936933858524724,
            "v1": 1.0,
            "v3": 0.0008681618870475666,
            "B": 0.46210742166589114,
        },
        covariance_matrix={
            ("ns_amplitude", "ns_amplitude"): 0.04846133253279189,
            ("ns_amplitude", "as_amplitude"): 0.02632121772890272,
            ("ns_amplitude", "ns_sigma"): 0.002252527350040729,
            ("ns_amplitude", "as_sigma"): 0.006995623419233939,
            ("ns_amplitude", "BG"): -0.011635567365839462,
            ("ns_amplitude", "v2_t"): -1.997103063002631e-09,
            ("ns_amplitude", "v2_a"): 2.3954482558423866e-07,
            ("ns_amplitude", "v4_t"): 2.5541716687991815e-11,
            ("ns_amplitude", "v4_a"): 2.667475073274938e-07,
            ("ns_amplitude", "v3"): -8.602836964339254e-05,
            ("ns_amplitude", "B"): -0.006702940356711055,
            ("as_amplitude", "ns_amplitude"): 0.02632121772890272,
            ("as_amplitude", "as_amplitude"): 0.0798964238443217,
            ("as_amplitude", "ns_sigma"): 0.0012367143527582136,
            ("as_amplitude", "as_sigma"): 0.013227317566298415,
            ("as_amplitude", "BG"): -0.01681898685960936,
            ("as_amplitude", "v2_t"): -2.3280579277848087e-09,
            ("as_amplitude", "v2_a"): -8.638378687247485e-07,
            ("as_amplitude", "v4_t"): 8.152906213924981e-10,
            ("as_amplitude", "v4_a"): 1.3429173761007637e-07,
            ("as_amplitude", "v3"): -5.1921742330894725e-05,
            ("as_amplitude", "B"): -0.004045983645905941,
            ("ns_sigma", "ns_amplitude"): 0.002252527350040729,
            ("ns_sigma", "as_amplitude"): 0.0012367143527582136,
            ("ns_sigma", "ns_sigma"): 0.00025056834280290586,
            ("ns_sigma", "as_sigma"): 0.0002364643220729578,
            ("ns_sigma", "BG"): -0.0005517133576386456,
            ("ns_sigma", "v2_t"): -8.635121192809922e-11,
            ("ns_sigma", "v2_a"): -3.9272492313043876e-08,
            ("ns_sigma", "v4_t"): 5.680557315666346e-11,
            ("ns_sigma", "v4_a"): 3.874027709281244e-09,
            ("ns_sigma", "v3"): -1.6200420139126602e-06,
            ("ns_sigma", "B"): -0.00012624705053034544,
            ("as_sigma", "ns_amplitude"): 0.006995623419233939,
            ("as_sigma", "as_amplitude"): 0.013227317566298415,
            ("as_sigma", "ns_sigma"): 0.0002364643220729578,
            ("as_sigma", "as_sigma"): 0.005295921830156029,
            ("as_sigma", "BG"): -0.0032049182346621497,
            ("as_sigma", "v2_t"): -1.530066506545061e-10,
            ("as_sigma", "v2_a"): 7.14502308354666e-07,
            ("as_sigma", "v4_t"): 5.95939607748049e-10,
            ("as_sigma", "v4_a"): 1.2846634840719582e-07,
            ("as_sigma", "v3"): -3.604332541105233e-05,
            ("as_sigma", "B"): -0.0028080429559457576,
            ("BG", "ns_amplitude"): -0.011635567365839462,
            ("BG", "as_amplitude"): -0.01681898685960936,
            ("BG", "ns_sigma"): -0.0005517133576386456,
            ("BG", "as_sigma"): -0.0032049182346621497,
            ("BG", "BG"): 0.007503753078738351,
            ("BG", "v2_t"): 6.89514178994482e-10,
            ("BG", "v2_a"): 1.0537517618602591e-07,
            ("BG", "v4_t"): -1.3647027909619434e-10,
            ("BG", "v4_a"): -6.302808557238417e-08,
            ("BG", "v3"): 2.1744348197413337e-05,
            ("BG", "B"): 0.0016942994655952252,
            ("v2_t", "ns_amplitude"): -1.997103063002631e-09,
            ("v2_t", "as_amplitude"): -2.3280579277848087e-09,
            ("v2_t", "ns_sigma"): -8.635121192809922e-11,
            ("v2_t", "as_sigma"): -1.530066506545061e-10,
            ("v2_t", "BG"): 6.89514178994482e-10,
            ("v2_t", "v2_t"): 1.6426141683899495e-10,
            ("v2_t", "v2_a"): 4.548222250576325e-12,
            ("v2_t", "v4_t"): -1.2118832422939818e-15,
            ("v2_t", "v4_a"): 8.335558954386748e-13,
            ("v2_t", "v3"): 1.4886768914860179e-12,
            ("v2_t", "B"): -1.4039293520415545e-09,
            ("v2_a", "ns_amplitude"): 2.3954482558423866e-07,
            ("v2_a", "as_amplitude"): -8.638378687247485e-07,
            ("v2_a", "ns_sigma"): -3.9272492313043876e-08,
            ("v2_a", "as_sigma"): 7.14502308354666e-07,
            ("v2_a", "BG"): 1.0537517618602591e-07,
            ("v2_a", "v2_t"): 4.548222250576325e-12,
            ("v2_a", "v2_a"): 7.474486187274642e-06,
            ("v2_a", "v4_t"): -1.075426302568744e-11,
            ("v2_a", "v4_a"): 2.0039178391235273e-07,
            ("v2_a", "v3"): -1.648108668028786e-08,
            ("v2_a", "B"): 2.3229894564596034e-06,
            ("v4_t", "ns_amplitude"): 2.5541716687991815e-11,
            ("v4_t", "as_amplitude"): 8.152906213924981e-10,
            ("v4_t", "ns_sigma"): 5.680557315666346e-11,
            ("v4_t", "as_sigma"): 5.95939607748049e-10,
            ("v4_t", "BG"): -1.3647027909619434e-10,
            ("v4_t", "v2_t"): -1.2118832422939818e-15,
            ("v4_t", "v2_a"): -1.075426302568744e-11,
            ("v4_t", "v4_t"): 1.7100610209014645e-09,
            ("v4_t", "v4_a"): -2.440735706023077e-11,
            ("v4_t", "v3"): -2.8635331971346696e-12,
            ("v4_t", "B"): -1.7517308704640773e-09,
            ("v4_a", "ns_amplitude"): 2.667475073274938e-07,
            ("v4_a", "as_amplitude"): 1.3429173761007637e-07,
            ("v4_a", "ns_sigma"): 3.874027709281244e-09,
            ("v4_a", "as_sigma"): 1.2846634840719582e-07,
            ("v4_a", "BG"): -6.302808557238417e-08,
            ("v4_a", "v2_t"): 8.335558954386748e-13,
            ("v4_a", "v2_a"): 2.0039178391235273e-07,
            ("v4_a", "v4_t"): -2.440735706023077e-11,
            ("v4_a", "v4_a"): 1.0211578282445497e-05,
            ("v4_a", "v3"): -2.7240320839964862e-09,
            ("v4_a", "B"): -4.637048315969915e-07,
            ("v3", "ns_amplitude"): -8.602836964339254e-05,
            ("v3", "as_amplitude"): -5.1921742330894725e-05,
            ("v3", "ns_sigma"): -1.6200420139126602e-06,
            ("v3", "as_sigma"): -3.604332541105233e-05,
            ("v3", "BG"): 2.1744348197413337e-05,
            ("v3", "v2_t"): 1.4886768914860179e-12,
            ("v3", "v2_a"): -1.648108668028786e-08,
            ("v3", "v4_t"): -2.8635331971346696e-12,
            ("v3", "v4_a"): -2.7240320839964862e-09,
            ("v3", "v3"): 7.538497097706797e-07,
            ("v3", "B"): 5.872976118430097e-05,
            ("B", "ns_amplitude"): -0.006702940356711055,
            ("B", "as_amplitude"): -0.004045983645905941,
            ("B", "ns_sigma"): -0.00012624705053034544,
            ("B", "as_sigma"): -0.0028080429559457576,
            ("B", "BG"): 0.0016942994655952252,
            ("B", "v2_t"): -1.4039293520415545e-09,
            ("B", "v2_a"): 2.3229894564596034e-06,
            ("B", "v4_t"): -1.7517308704640773e-09,
            ("B", "v4_a"): -4.637048315969915e-07,
            ("B", "v3"): 5.872976118430097e-05,
            ("B", "B"): 0.21354337397198286,
        },
        errors=[],
        x=np.array([
            -1.48352986, -1.30899694, -1.13446401, -0.95993109, -0.78539816, -0.61086524,
            -0.43633231, -0.26179939, -0.08726646, 0.08726646, 0.26179939, 0.43633231,
            0.61086524, 0.78539816, 0.95993109, 1.13446401, 1.30899694, 1.48352986,
            1.65806279, 1.83259571, 2.00712864, 2.18166156, 2.35619449, 2.53072742,
            2.70526034, 2.87979327, 3.05432619, 3.22885912, 3.40339204, 3.57792497,
            3.75245789, 3.92699082, 4.10152374, 4.27605667, 4.45058959, 4.62512252,
        ]),
        n_fit_data_points=90,
        minimum_val=129.53525910258875,
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
    expected_components = {
        base.FitType(region='signal', orientation='inclusive'): base.BaseFitResult(
            parameters = expected_signal_parameters,
            free_parameters = expected_signal_free_parameters,
            fixed_parameters = ['signal_pedestal', 'v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_signal_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_signal_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_signal_free_parameters for k2 in expected_signal_free_parameters},
            errors = np.array([
                0.08281593, 0.09514459, 0.11058085, 0.11038697, 0.09246798, 0.08607996,
                0.18658103, 0.19248293, 0.25152553, 0.25152553, 0.19248293, 0.18658103,
                0.08607986, 0.09246791, 0.11039063, 0.11059964, 0.09510031, 0.08156646,
                0.09114142, 0.11009599, 0.11508615, 0.10894117, 0.11631684, 0.12684909,
                0.11718967, 0.13390741, 0.1801032 , 0.1801032 , 0.13390741, 0.11718967,  # noqa: E203
                0.12684909, 0.11631684, 0.10894117, 0.11508615, 0.11009599, 0.09114142,
            ]),
        ),
        base.FitType(region='background', orientation='in_plane'): base.BaseFitResult(
            parameters = expected_background_parameters,
            free_parameters=expected_background_free_parameters,
            fixed_parameters=['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            errors = np.array([
                0.32841373, 0.28401352, 0.23776085, 0.22430129, 0.22249343, 0.2152592 ,  # noqa: E203
                0.24073382, 0.30957262, 0.3659735 , 0.3659735 , 0.30957262, 0.24073382,  # noqa: E203
                0.2152592 , 0.22249343, 0.22430129, 0.23776085, 0.28401352, 0.32841373,  # noqa: E203
                0.33221576, 0.29585571, 0.25675   , 0.24435686, 0.23745899, 0.22103222,  # noqa: E203
                0.23544904, 0.2982731 , 0.35293121, 0.35293121, 0.2982731 , 0.23544904,  # noqa: E203
                0.22103222, 0.23745899, 0.24435686, 0.25675   , 0.29585571, 0.33221576,  # noqa: E203
            ])
        ),
        base.FitType(region='background', orientation='mid_plane'): base.BaseFitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            errors = np.array([
                0.32813719, 0.22211081, 0.17817623, 0.29246334, 0.34944401, 0.28274342,
                0.16931736, 0.23705456, 0.35018994, 0.35018994, 0.23705456, 0.16931736,
                0.28274342, 0.34944401, 0.29246334, 0.17817623, 0.22211081, 0.32813719,
                0.3319468 , 0.23705309, 0.20273059, 0.30801095, 0.35908298, 0.28713307,  # noqa: E203
                0.16174752, 0.22211238, 0.33652361, 0.33652361, 0.22211238, 0.16174752,  # noqa: E203
                0.28713307, 0.35908298, 0.30801095, 0.20273059, 0.23705309, 0.3319468 ,  # noqa: E203
            ])
        ),
        base.FitType(region='background', orientation='out_of_plane'): base.BaseFitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            errors = np.array([
                0.34368969, 0.29725271, 0.24622698, 0.22709399, 0.22168468, 0.21070458,
                0.23040181, 0.29484674, 0.34926328, 0.34926328, 0.29484674, 0.23040181,
                0.21070458, 0.22168468, 0.22709399, 0.24622698, 0.29725271, 0.34368969,
                0.34730151, 0.30851602, 0.2644947 , 0.24679856, 0.23660525, 0.21656012,  # noqa: E203
                0.22491218, 0.28304249, 0.33566777, 0.33566777, 0.28304249, 0.22491218,
                0.21656012, 0.23660525, 0.24679856, 0.2644947 , 0.30851602, 0.34730151,  # noqa: E203
            ])
        ),
    }

    # Run the fit
    # NOTE: The user_arguments are the same as the defaults to ensure that they don't change the fit, but we specify
    #       one to test the logic of how they are set.
    rp_fit, full_componnents, data, minuit = example.run_inclusive_signal_fit(
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
    fig, ax = plot.draw_fit(rp_fit = rp_fit, data = data, fit_label = "Inclusive signal RP fit", filename = "")
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
        parameters=["B", "v2_t", "v2_a", "v4_t", "v4_a", "v1", "v3"],
        free_parameters=["B", "v2_t", "v2_a", "v4_t", "v4_a", "v3"],
        fixed_parameters=["v1"],
        values_at_minimum={
            "B": 176.33321411648893,
            "v2_t": 0.0010001897811286388,
            "v2_a": 0.04669637567679903,
            "v4_t": 4.7683700662992656e-07,
            "v4_a": 0.005909769311241236,
            "v1": 0.0,
            "v3": 0.007017169497204351,
        },
        errors_on_parameters={
            "B": 0.48082748291253097,
            "v2_t": 0.00020980194133389562,
            "v2_a": 0.002728782407465522,
            "v4_t": 0.000921883930829856,
            "v4_a": 0.003185271859029795,
            "v1": 1.0,
            "v3": 0.0019404574651313428,
        },
        covariance_matrix={
            ("B", "B"): 0.23119519099746746,
            ("B", "v2_t"): -1.4647482573912556e-09,
            ("B", "v2_a"): -3.0751892773390162e-06,
            ("B", "v4_t"): -1.560004669861541e-09,
            ("B", "v4_a"): -1.3288172060078484e-06,
            ("B", "v3"): 0.00028415169669183536,
            ("v2_t", "B"): -1.4647482573912556e-09,
            ("v2_t", "v2_t"): 1.5932195071257722e-10,
            ("v2_t", "v2_a"): 3.5221606052087055e-14,
            ("v2_t", "v4_t"): -1.2505615081980165e-15,
            ("v2_t", "v4_a"): 7.127938489632922e-13,
            ("v2_t", "v3"): 1.5508494510732615e-13,
            ("v2_a", "B"): -3.0751892773390162e-06,
            ("v2_a", "v2_t"): 3.5221606052087055e-14,
            ("v2_a", "v2_a"): 7.4486966943423805e-06,
            ("v2_a", "v4_t"): -1.0504595437166638e-11,
            ("v2_a", "v4_a"): 1.9958812649761715e-07,
            ("v2_a", "v3"): -8.911618790044292e-08,
            ("v4_t", "B"): -1.560004669861541e-09,
            ("v4_t", "v2_t"): -1.2505615081980165e-15,
            ("v4_t", "v2_a"): -1.0504595437166638e-11,
            ("v4_t", "v4_t"): 1.7594369052582678e-09,
            ("v4_t", "v4_a"): 4.18443971580955e-12,
            ("v4_t", "v3"): 2.2326619374354244e-13,
            ("v4_a", "B"): -1.3288172060078484e-06,
            ("v4_a", "v2_t"): 7.127938489632922e-13,
            ("v4_a", "v2_a"): 1.9958812649761715e-07,
            ("v4_a", "v4_t"): 4.18443971580955e-12,
            ("v4_a", "v4_a"): 1.0157729997549444e-05,
            ("v4_a", "v3"): -1.3797698330732505e-08,
            ("v3", "B"): 0.00028415169669183536,
            ("v3", "v2_t"): 1.5508494510732615e-13,
            ("v3", "v2_a"): -8.911618790044292e-08,
            ("v3", "v4_t"): 2.2326619374354244e-13,
            ("v3", "v4_a"): -1.3797698330732505e-08,
            ("v3", "v3"): 3.7667421273908302e-06,
        },
        errors=[],
        x=np.array([
            -1.48352986, -1.30899694, -1.13446401, -0.95993109, -0.78539816, -0.61086524,
            -0.43633231, -0.26179939, -0.08726646, 0.08726646, 0.26179939, 0.43633231,
            0.61086524, 0.78539816, 0.95993109, 1.13446401, 1.30899694, 1.48352986,
            1.65806279, 1.83259571, 2.00712864, 2.18166156, 2.35619449, 2.53072742,
            2.70526034, 2.87979327, 3.05432619, 3.22885912, 3.40339204, 3.57792497,
            3.75245789, 3.92699082, 4.10152374, 4.27605667, 4.45058959, 4.62512252,
        ]),
        n_fit_data_points=54,
        minimum_val=84.21205912913777,
    )
    expected_background_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v1', 'v3']
    expected_background_free_parameters = ['B', 'v2_t', 'v2_a', 'v4_t', 'v4_a', 'v3']
    expected_components = {
        base.FitType(region='background', orientation='in_plane'): base.BaseFitResult(
            parameters = expected_background_parameters,
            free_parameters=expected_background_free_parameters,
            fixed_parameters=['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            errors = np.array([
                0.3284827, 0.30117424, 0.28296194, 0.27178786, 0.24412456, 0.21539993,
                0.25957565, 0.36262685, 0.43863545, 0.43863545, 0.36262685, 0.25957565,
                0.21539993, 0.24412456, 0.27178786, 0.28296194, 0.30117424, 0.3284827,
                0.34634315, 0.35159443, 0.35351492, 0.34448918, 0.30378086, 0.24147163,
                0.23563421, 0.31501509, 0.38526347, 0.38526347, 0.31501509, 0.23563421,
                0.24147163, 0.30378086, 0.34448918, 0.35351492, 0.35159443, 0.34634315,
            ])
        ),
        base.FitType(region='background', orientation='mid_plane'): base.BaseFitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            errors = np.array([
                0.32806042, 0.24370903, 0.23525068, 0.33028055, 0.36348313, 0.28273384,
                0.1952621, 0.30327089, 0.42572494, 0.42572494, 0.30327089, 0.1952621,
                0.28273384, 0.36348313, 0.33028055, 0.23525068, 0.24370903, 0.32806042,
                0.34578768, 0.30326988, 0.3157949, 0.39161668, 0.40559043, 0.3029897,
                0.16197158, 0.24371055, 0.36967661, 0.36967661, 0.24371055, 0.16197158,
                0.3029897, 0.40559043, 0.39161668, 0.3157949, 0.30326988, 0.34578768,
            ])
        ),
        base.FitType(region='background', orientation='out_of_plane'): base.BaseFitResult(
            parameters = expected_background_parameters,
            free_parameters = expected_background_free_parameters,
            fixed_parameters = ['v1'],
            values_at_minimum = {k: expected_fit_result.values_at_minimum[k] for k in expected_background_parameters},
            errors_on_parameters = {
                k: expected_fit_result.errors_on_parameters[k] for k in expected_background_parameters
            },
            covariance_matrix = {(k1, k2): expected_fit_result.covariance_matrix[(k1, k2)] for k1 in expected_background_free_parameters for k2 in expected_background_free_parameters},
            errors = np.array([
                0.34378357, 0.31394143, 0.29028637, 0.27406474, 0.24325095, 0.21082366,
                0.25021672, 0.35040625, 0.42506549, 0.42506549, 0.35040625, 0.25021672,
                0.21082366, 0.24325095, 0.27406474, 0.29028637, 0.31394143, 0.34378357,
                0.36040536, 0.36140036, 0.35808058, 0.34535821, 0.30272333, 0.23742926,
                0.2250676, 0.30014813, 0.36877646, 0.36877646, 0.30014813, 0.2250676,
                0.23742926, 0.30272333, 0.34535821, 0.35808058, 0.36140036, 0.36040536,
            ])
        ),
    }

    # Run the fit
    # NOTE: The user_arguments are the same as the defaults to ensure that they don't change the fit, but we specify
    #       one to test the logic of how they are set.
    rp_fit, full_componnents, data, minuit = example.run_background_fit(
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
    summary_components = rp_fit.create_full_set_of_components(input_data = data)

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
    fig, ax = plot.draw_residual(rp_fit = rp_fit, data = data, fit_label = "Standard RP fit", filename = "")
    return fig

@pytest.mark.slow  # type: ignore
@pytest.mark.parametrize("calculate_correlation_matrix_before_writing", [  # type: ignore
    False, True,
], ids = ["Don't calculate correlation matrix", "Calculate correlation matrix"])
@pytest.mark.parametrize("example_module_func, fit_object", [  # type: ignore
    (example.run_background_fit, three_orientations.BackgroundFit),
    (example.run_inclusive_signal_fit, three_orientations.InclusiveSignalFit),
], ids = ["Background", "Inclusive signal"])
def test_write_and_read_result_in_class(logging_mixin: Any, setup_integration_tests: Any,
                                        calculate_correlation_matrix_before_writing: bool,
                                        example_module_func: Any, fit_object: Any) -> None:
    """ Test storing results via the pachyderm `yaml` module.

    Note:
        This ends up as an integration test because we actually use the yaml module
        to read and write to a file instead of mocking it.
    """
    # Setup
    sample_data_filename = setup_integration_tests

    expected_rp_fit, full_componnents, data, minuit = example_module_func(
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

