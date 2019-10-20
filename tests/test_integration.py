#!/usr/bin/env python

""" Integration tests for the inclusive signal and background only fits.

The tests are performed by leveraging the example fit code, as it runs an entire fit.

By using integration tests, we can tests large positions of the package more rapidly.
This of course trades granularity and may allow some branches to slip through, but the
time savings is worth it.

Note:
    These tests are not necessarily representative of good fits - they are just what you
    get when you fit with default values. That means these will change if the default values
    are modified!

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import pkg_resources
import pytest
import tempfile
from pathlib import Path
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
        atol = 0, rtol = 5e-3,
    )
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.errors_on_parameters.keys()) == list(expected_fit_result.errors_on_parameters.keys())
    # Need the extra tolerance to work on other systems.
    np.testing.assert_allclose(
        list(fit_result.errors_on_parameters.values()),
        list(expected_fit_result.errors_on_parameters.values()),
        atol = 1e-2, rtol = 0
    )
    # Need to compare separately the keys and values so we can use np.allclose() for the values
    assert list(fit_result.covariance_matrix.keys()) == list(expected_fit_result.covariance_matrix.keys())
    # Need the extra tolerance to work on other systems.
    np.testing.assert_allclose(
        list(fit_result.covariance_matrix.values()),
        list(expected_fit_result.covariance_matrix.values()),
        atol = 0, rtol = 5e-2,
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
@pytest.mark.mpl_image_compare(tolerance = 6.5)  # type: ignore
def test_inclusive_signal_fit(setup_integration_tests: Any) -> Figure:
    """ Integration test for the inclusive signal fit.

    This uses the sample data in the ``sample_data`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    # NOTE: They are not calculated independently (they use the same method, just at a particular time), so there
    #       are more like regression tests.
    expected_rp_fit = three_orientations.InclusiveSignalFit(
        resolution_parameters = {"R22": 1, "R42": 1, "R62": 1, "R82": 1},
        use_log_likelihood = False,
        signal_region = (0, 0.6),
        background_region = (0.8, 1.2),
    )
    expected_yaml_filename = Path(__file__).parent / "testFiles" / "expected_inclusive_signal_fit.yaml"
    expected_rp_fit.read_fit_results(str(expected_yaml_filename))
    expected_fit_result = expected_rp_fit.fit_result

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
        assert compare_fit_result_to_expected(
            fit_result = fit_component.fit_result,
            expected_fit_result = expected_rp_fit.components[fit_type].fit_result
        ) is True

    # Draw and check the resulting image. It is checked by returning the figure.
    # We skip the residual plot because it is hard to compare multiple images from in the same test.
    # Instead, we get around this by comparing the residual in test_background_fit(...)
    fig, ax = plot.draw_fit(rp_fit = rp_fit, data = data, fit_label = "Inclusive signal RP fit", filename = "")
    return fig

@pytest.mark.slow  # type: ignore
@pytest.mark.mpl_image_compare(tolerance = 6.5)  # type: ignore
def test_differential_signal_fit(setup_integration_tests: Any) -> Figure:
    """ Integration test for the differential signal fit.

    This uses the sample data in the ``sample_data`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    # NOTE: They are not calculated independently (they use the same method, just at a particular time), so there
    #       are more like regression tests.
    expected_rp_fit = three_orientations.SignalFit(
        resolution_parameters = {"R22": 1, "R42": 1, "R62": 1, "R82": 1},
        use_log_likelihood = False,
        signal_region = (0, 0.6),
        background_region = (0.8, 1.2),
    )
    expected_yaml_filename = Path(__file__).parent / "testFiles" / "expected_differential_signal_fit.yaml"
    expected_rp_fit.read_fit_results(str(expected_yaml_filename))
    expected_fit_result = expected_rp_fit.fit_result

    # Run the fit
    # NOTE: The user_arguments are the same as the defaults to ensure that they don't change the fit, but we specify
    #       one to test the logic of how they are set.
    rp_fit, full_componnents, data, minuit = example.run_differential_signal_fit(
        input_filename = sample_data_filename,
        user_arguments = {"v2_t": 0.02},
    )

    # Check the result
    assert compare_fit_result_to_expected(fit_result = rp_fit.fit_result,
                                          expected_fit_result = expected_fit_result,
                                          minuit = minuit) is True
    # Check the components
    for fit_type, fit_component in rp_fit.components.items():
        assert compare_fit_result_to_expected(
            fit_result = fit_component.fit_result,
            expected_fit_result = expected_rp_fit.components[fit_type].fit_result
        ) is True

    # Draw and check the resulting image. It is checked by returning the figure.
    # We skip the residual plot because it is hard to compare multiple images from in the same test.
    # Instead, we get around this by comparing the residual in test_background_fit(...)
    fig, ax = plot.draw_fit(rp_fit = rp_fit, data = data, fit_label = "Differential signal RP fit", filename = "")
    return fig

@pytest.mark.slow  # type: ignore
@pytest.mark.mpl_image_compare(tolerance = 6.5)  # type: ignore
def test_background_fit(setup_integration_tests: Any) -> Figure:
    """ Integration test for the background fit.

    This uses the sample data in the ``sample_data`` directory.
    """
    sample_data_filename = setup_integration_tests
    # Setup the expected fit result. These values are extracted from an example fit.
    # NOTE: They are not calculated independently (they use the same method, just at a particular time), so there
    #       are more like regression tests.
    expected_rp_fit = three_orientations.BackgroundFit(
        resolution_parameters = {"R22": 1, "R42": 1, "R62": 1, "R82": 1},
        use_log_likelihood = False,
        signal_region = (0, 0.6),
        background_region = (0.8, 1.2),
    )
    expected_yaml_filename = Path(__file__).parent / "testFiles" / "expected_background_fit.yaml"
    expected_rp_fit.read_fit_results(str(expected_yaml_filename))
    expected_fit_result = expected_rp_fit.fit_result

    # Run the fit for comparison
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
        assert compare_fit_result_to_expected(
            fit_result = fit_component.fit_result,
            expected_fit_result = expected_rp_fit.components[fit_type].fit_result
        ) is True

    # Check standard summary components
    # Create the full set of components and check them out.
    summary_components = rp_fit.create_full_set_of_components(input_data = data)

    # Make sure that we have all of the components (including the inclusive)
    assert len(summary_components) == len(rp_fit._rp_orientations)

    # Check the inclusive summary component.
    inclusive_component = summary_components["inclusive"]
    assert inclusive_component.fit_function == three_orientations.unconstrained_inclusive_background
    assert inclusive_component.fit_result.parameters == expected_fit_result.parameters
    # Check values
    values_at_minimum = expected_fit_result.values_at_minimum
    assert list(inclusive_component.fit_result.values_at_minimum) == list(values_at_minimum)
    np.testing.assert_allclose(
        list(inclusive_component.fit_result.values_at_minimum.values()), list(values_at_minimum.values()),
        atol = 2e-4, rtol = 0,
    )
    # We want to compare against the Fourier values.
    #values_at_minimum["B"] = values_at_minimum["B"] / 3
    x = expected_fit_result.x
    expected_inclusive_component_values = functions.fourier(x, *list(values_at_minimum.values()))
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

