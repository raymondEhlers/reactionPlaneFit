#!/usr/bin/env python

""" Example of how to use the Reaction Plane Fit package.

Provides example for fitting only the background, or fitting the inclusive signal.

.. code-author: Raymond Ehlers <raymond.ehlers@ycern.ch>, Yale University
"""

import argparse
import logging
import pkg_resources
from typing import Tuple, Type, TYPE_CHECKING
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
FitReturnValues = Tuple[fit.ReactionPlaneFit, Data, "iminuit.Minuit"]

def setup_data(input_filename: str, include_signal: bool) -> InputData:
    """ Setup the example input data.

    Read the data using uproot so we can avoid an explicit dependency on ROOT. Histograms
    are assumed to be named ``{region}_{orientation}`` (for example, "signalDominated_inclusive").

    Args:
        input_filename (str): Path to the input data to use.
        include_signal (bool): If true, the signal will be included in the data dict.
    Returns:
        dict: Containing the input data.
    """
    data: dict = {"signal": {}, "background": {}}
    with uproot.open(input_filename) as f:
        if include_signal:
            data["signal"]["inclusive"] = f["signalDominated_inclusive"]
        for rp in ["in_plane", "mid_plane", "out_of_plane"]:
            data["background"][rp] = f[f"backgroundDominated_{rp}"]

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
        tuple: (rp_fit, data), where rp_fit (fit.ReactionPlaneFit) is the reaction plane fit object from the
            fit, and data (dict) is the formated data dict used for the fit.
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

    if success:
        logger.info(f"Fit was successful! Fit result: {rp_fit.fit_result}")

    return rp_fit, data, minuit

def run_background_fit(input_filename: str, user_arguments: FitArguments) -> FitReturnValues:
    """ Run the background example fit.

    Args:
        input_filename: Path to the input data to use.
        user_arguments: User arguments to override the arguments to the fit.
    Returns:
        tuple: (rp_fit, data), where rp_fit (fit.ReactionPlaneFit) is the reaction plane fit object from the
            fit, and data (dict) is the formated data dict used for the fit.
    """
    # Grab the input data.
    input_data = setup_data(input_filename, include_signal = False)
    rp_fit, data, minuit = run_fit(
        fit_object = three_orientations.BackgroundFit,
        input_data = input_data, user_arguments = user_arguments
    )
    return rp_fit, data, minuit

def run_inclusive_signal_fit(input_filename: str, user_arguments: FitArguments) -> FitReturnValues:
    """ Run the inclusive signal example fit.

    Args:
        input_filename: Path to the input data to use.
        user_arguments: User arguments to override the arguments to the fit.
    Returns:
        tuple: (rp_fit, data), where rp_fit (fit.ReactionPlaneFit) is the reaction plane fit object from the
            fit, and data (dict) is the formated data dict used for the fit.
    """
    input_data = setup_data(input_filename, include_signal = True)
    rp_fit, data, minuit = run_fit(
        fit_object = three_orientations.InclusiveSignalFit,
        input_data = input_data, user_arguments = user_arguments
    )
    return rp_fit, data, minuit

if __name__ == "__main__":  # pragma: nocover
    """ Allow direction execution of this module.

    The user can specify the input filename and which type of fit to perform. However, the names of the input
    histograms must be as specified in ``setup_data(...)``.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    # Setup parser
    parser = argparse.ArgumentParser(description = "Example Reaction Plane Fit using signal and background dominated sample data.")

    sample_data_filename = pkg_resources.resource_filename("reaction_plane_fit.sample_data", "three_orientations.root")
    # Set the input filename
    parser.add_argument("-i", "--inputData", metavar = "filename",
                        type = str, default = sample_data_filename,
                        help="Path to input data")
    # Set the fit type
    parser.add_argument("-b", "--backgroundOnly",
                        action = "store_true",
                        help = "Only fit the background.")
    # Parse arguments
    args = parser.parse_args()

    # Execute the selected function
    func = run_inclusive_signal_fit
    if args.backgroundOnly:
        func = run_background_fit
    rp_fit, data, _ = func(input_filename = args.inputData, user_arguments = {})

    # Draw the plots so that they will be saved out.
    plot.draw_fit(rp_fit = rp_fit, data = data, filename = "example_fit.png")
    plot.draw_residual(rp_fit = rp_fit, data = data, filename = "example_residual.png")

