#!/usr/bin/env python

""" Example of how to use the Reaction Plane Fit package.

.. code-author: Raymond Ehlers <raymond.ehlers@ycern.ch>, Yale University
"""

import argparse
import logging
import os
import uproot

from reactionPlaneFit import fit
from reactionPlaneFit import three_orientations

logger = logging.getLogger(__name__)

def setup_data(input_filename: str) -> dict:
    """ Setup the example input data.

    Read the data using uproot so we can avoid an explicit dependency on ROOT.

    Args:
        input_filename (str): Path to the input data to use.
    Returns:
        dict: Containing the input data.
    """
    data = {"signal": {}, "background": {}}
    with uproot.open(input_filename) as f:
        data["signal"]["inclusive"] = f["signalDominated_inclusive"]
        for rp in ["inPlane", "midPlane", "outOfPlane"]:
            data["background"][rp] = f[f"backgroundDominated_{rp}"]

    #x = np.linspace(-np.pi / 2, 3. / 2. * np.pi, 36 + 1)
    ## Gaussian signal
    #ns_signal = probfit.pdf.gaussian(x, mean = 0, sigma = 0.5)
    #as_signal = probfit.pdf.gaussian(x, mean = np.pi, sigma = 0.7)
    #bg_amplitude = 3.0
    ## Background data.
    #data["inPlane"] = ns_signal + as_signal + bg_amplitude * np.cos(2*x)
    #data["midPlane"] = ns_signal + as_signal + bg_amplitude * np.cos(3*x)
    #data["inPlane"] = ns_signal + as_signal - bg_amplitude * np.cos(2*x)

    return data

def run_fit(input_filename: str) -> fit.ReactionPlaneFit:
    """ Driver function for performing the fit.

    Note:
        We perform a fit including the signal for demonstration purposes. This is also an object
        for fitting just the background.

    Args:
        input_filename (str): Path to the input data to use.
    Returns:
        fit.ReactionPlaneFit: The Reaction Plane Fit object from the fit.
    """
    # Grab the input data.
    data = setup_data(input_filename)

    # Define the fit.
    rp_fit = three_orientations.InclusiveSignalFit(
        resolutionParameters = {"R22": 1, "R42": 1, "R62": 1, "R82": 1},
        useLogLikelihood = False,
        signalRegion = (0, 0.6),
        backgroundRegion = (0.8, 1.2),
    )

    # Perform the actual fit.
    fit_result = rp_fit.fit(data = data)

    if fit_result:
        logger.info("Fit was successful! Fit result: {fit_result}")

    return fit_result

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    # Setup parser
    parser = argparse.ArgumentParser(description = "Example Reaction Plane Fit using signal and background dominated sample data.")
    sample_data_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests", "testFiles", "sampleData.root")
    # Only one option
    parser.add_argument("-i", "--inputData", metavar="filename",
                        type = str, default = sample_data_filename,
                        help="Path to input data")
    # Parse arguments
    args = parser.parse_args()

    run_fit(input_filename = args.inputData)

