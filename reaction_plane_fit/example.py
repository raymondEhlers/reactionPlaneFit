#!/usr/bin/env python

""" Example of how to use the Reaction Plane Fit package.

Provides example for fitting only the background, or fitting the inclusive signal.

.. code-author: Raymond Ehlers <raymond.ehlers@ycern.ch>, Yale University
"""

import argparse
import logging
import pkg_resources
from typing import Any, Dict, Tuple, Type, TYPE_CHECKING
import uproot

from reaction_plane_fit import fit
from reaction_plane_fit.fit import Data, InputData
from reaction_plane_fit.fit import FitArguments
from reaction_plane_fit import three_orientations
from reaction_plane_fit import plot

if TYPE_CHECKING:
    import iminuit  # noqa: F401

logger = logging.getLogger(__name__)

# Type helpers
FitReturnValues = Tuple[fit.ReactionPlaneFit, Dict[str, fit.FitComponent], Data, "iminuit.Minuit"]

def setup_data(input_filename: str) -> InputData:
    """ Setup the example input data.

    Read the data using uproot so we can avoid an explicit dependency on ROOT. Histograms
    are assumed to be named ``{region}_{orientation}`` (for example, "signalDominated_inclusive").

    Args:
        input_filename: Path to the input data to use.
    Returns:
        dict: Containing the input data.
    """
    data: Dict[str, Any] = {"signal": {}, "background": {}}
    with uproot.open(input_filename) as f:
        for region in ["signal", "background"]:
            for rp in ["inclusive", "in_plane", "mid_plane", "out_of_plane"]:
                data[region][rp] = f[f"{region}Dominated_{rp}"]

    return data

def run_fit(fit_object: Type[fit.ReactionPlaneFit],
            input_data: InputData,
            user_arguments: FitArguments) -> FitReturnValues:
    """ Driver function for performing the fit.

    Note:
        Here we just set the resolution parameters to 1 for convenience of the example, but they are an extremely
        important part of the fit!

    Args:
        fit_object: Fit object to be used.
        data: Input data for the fit, labeled as defined in ``setup_data()``.
        user_arguments: User arguments to override the arguments to the fit.
    Returns:
        tuple: (rp_fit, full_components, data, minuit), where rp_fit (fit.ReactionPlaneFit) is the reaction plane
            fit object from the fit, full_components is the full set of fit components (one for each RP orientation),
            data (dict) is the formated data dict used for the fit, and minuit is the minuit fit object.
    """
    # Define the fit.
    rp_fit = fit_object(
        resolution_parameters = {"R22": 1, "R42": 1, "R62": 1, "R82": 1},
        use_log_likelihood = False,
        signal_region = (0, 0.6),
        background_region = (0.8, 1.2),
    )

    # Perform the actual fit.
    success, data, minuit = rp_fit.fit(data = input_data, user_arguments = user_arguments)
    # Create the full set of fit components
    full_components = rp_fit.create_full_set_of_components(input_data = data)

    if success:
        logger.info(f"Fit was successful! Fit result: {rp_fit.fit_result}")

    return rp_fit, full_components, data, minuit

def run_background_fit(input_filename: str, user_arguments: FitArguments) -> FitReturnValues:
    """ Run the background example fit.

    Args:
        input_filename: Path to the input data to use.
        user_arguments: User arguments to override the arguments to the fit.
    Returns:
        tuple: (rp_fit, full_components, data, minuit), where rp_fit (fit.ReactionPlaneFit) is the reaction plane
            fit object from the fit, full_components is the full set of fit components (one for each RP orientation),
            data (dict) is the formated data dict used for the fit, and minuit is the minuit fit object.
    """
    # Grab the input data.
    input_data = setup_data(input_filename)
    return_values = run_fit(
        fit_object = three_orientations.BackgroundFit,
        input_data = input_data, user_arguments = user_arguments
    )
    return return_values

def run_inclusive_signal_fit(input_filename: str, user_arguments: FitArguments) -> FitReturnValues:
    """ Run the inclusive signal example fit.

    Args:
        input_filename: Path to the input data to use.
        user_arguments: User arguments to override the arguments to the fit.
    Returns:
        tuple: (rp_fit, full_components, data, minuit), where rp_fit (fit.ReactionPlaneFit) is the reaction plane
            fit object from the fit, full_components is the full set of fit components (one for each RP orientation),
            data (dict) is the formated data dict used for the fit, and minuit is the minuit fit object.
    """
    input_data = setup_data(input_filename)
    return_values = run_fit(
        fit_object = three_orientations.InclusiveSignalFit,
        input_data = input_data, user_arguments = user_arguments
    )
    return return_values

def run_differential_signal_fit(input_filename: str, user_arguments: FitArguments) -> FitReturnValues:
    """ Run the differential signal example fit.

    Args:
        input_filename: Path to the input data to use.
        user_arguments: User arguments to override the arguments to the fit.
    Returns:
        tuple: (rp_fit, full_components, data, minuit), where rp_fit (fit.ReactionPlaneFit) is the reaction plane
            fit object from the fit, full_components is the full set of fit components (one for each RP orientation),
            data (dict) is the formated data dict used for the fit, and minuit is the minuit fit object.
    """
    input_data = setup_data(input_filename)
    return_values = run_fit(
        fit_object = three_orientations.SignalFit,
        input_data = input_data, user_arguments = user_arguments
    )
    return return_values

if __name__ == "__main__":  # pragma: nocover
    """ Allow direction execution of this module.

    The user can specify the input filename and which type of fit to perform. However, the names of the input
    histograms must be as specified in ``setup_data(...)``.
    """
    # Setup logging
    logging.basicConfig(level = logging.INFO)
    # Quiet down the matplotlib logging regardless of the base logging level.
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # Setup parser
    parser = argparse.ArgumentParser(
        description = "Example Reaction Plane Fit using signal and background dominated sample data."
    )

    sample_data_filename = pkg_resources.resource_filename("reaction_plane_fit.sample_data", "three_orientations.root")
    # Set the input filename
    parser.add_argument("-i", "--inputData", metavar = "filename",
                        type = str, default = sample_data_filename,
                        help="Path to input data")
    # Set the fit type (defaults to differential signal fit)
    parser.add_argument("-d", "--differentialSignal",
                        action = "store_true",
                        help = "Fit the differential signal regions.")
    parser.add_argument("-b", "--backgroundOnly",
                        action = "store_true",
                        help = "Only fit the background.")
    # Parse arguments
    args = parser.parse_args()

    # Execute the selected function
    func = run_inclusive_signal_fit
    if args.differentialSignal:
        func = run_differential_signal_fit
    if args.backgroundOnly:
        func = run_background_fit
    rp_fit, _, data, _ = func(
        input_filename = args.inputData,
        user_arguments = {},
    )
    # Save to YAML.
    rp_fit.write_fit_results("example.yaml")

    # Determine fit label
    fit_label_map = {
        three_orientations.BackgroundFit: "Standard RP fit",
        three_orientations.InclusiveSignalFit: "Inclusive signal RP fit",
        three_orientations.SignalFit: "Differential signal RP fit",
    }
    fit_label = fit_label_map[type(rp_fit)]

    # Draw the plots so that they will be saved out.
    plot.draw_fit(rp_fit = rp_fit, data = data, fit_label = fit_label, filename = "example_fit.png")
    plot.draw_residual(rp_fit = rp_fit, data = data, fit_label = fit_label, filename = "example_residual.png")

