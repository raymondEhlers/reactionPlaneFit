#!/usr/bin/env python

""" Plotting helper functions related to the Reaction Plane Fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import numpy as np
from re import finditer
from typing import Callable

from reaction_plane_fit import fit

def camel_case_split(string: str) -> list:
    """ Helper function to split camel case into the constituent letters

    See: https://stackoverflow.com/a/29920015

    Args:
        string: Camel case string to be split.
    Returns:
        String split on each word.
    """
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", string)
    return [m.group(0) for m in matches]

def format_rp_labels(string: str) -> str:
    """ Helper function for formatting the RP label.

    Args:
        string: Camel case string to be split.
    Returns:
        Properly formatted RP label.
    """
    # Split string
    strings = camel_case_split(string)
    # Capitalize first word, and then lower case all of the rest.
    strings = [strings[0].capitalize()] + strings[1:]
    strings = strings[:1] + [s.lower() for s in strings[1:]]
    # Finally, join with dashes.
    return "-".join(strings)

def draw(rp_fit: fit.ReactionPlaneFit, data: dict, filename: str, y_label: str, draw_func: Callable[..., np.ndarray]):
    """ Main RP fit drawing function.

    It takes a callable to do the actual extraction of the relevant data and the plotting. This way, we can avoid repeating
    ourselves for mostly identical plots that include slightly different data.

    Args:
        rp_fit: Reaction plane fit object after fitting.
        data: Data used to perform the fit.
        filename: Location where the file should be saved.
        y_label: Label for the y-axis.
        draw_func: Function to take the given data and draw the necessary plots on the given axis.
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which everything has
            been drawn, as well as the axes used for drawing.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})

    # Setup
    x = rp_fit.fit_result.x
    n_components = len(rp_fit.components)
    fig, axes = plt.subplots(1, n_components, sharey = True, sharex = True, figsize = (3 * n_components, 6))

    # Shared y axis label
    axes[0].set_ylabel(y_label)

    for (fit_type, component), ax in zip(rp_fit.components.items(), axes):
        # Get the relevant data
        hist = data[fit_type]

        # Draw the data according to the given function
        draw_func(rp_fit = rp_fit, fit_type = fit_type, x = x, hist = hist, ax = ax)

        # Add axis labels
        ax.set_xlabel(r"$\Delta\varphi$")

        # Add fit_type labels
        text = f"{fit_type.region.capitalize()} dominated"
        text += f"\n" + format_rp_labels(fit_type.orientation)
        # Add the chi2/ndf for in the last panel.
        if ax == axes[-1]:
            text += "\n" + r"$\chi^{2}$/NDF = %(chi2).1f/%(ndf)i = %(chi2_over_ndf).3f" % {"chi2": rp_fit.fit_result.minimum_val, "ndf": rp_fit.fit_result.nDOF, "chi2_over_ndf": rp_fit.fit_result.minimum_val / rp_fit.fit_result.nDOF}
        ax.text(0.5, 0.95, text,
                horizontalalignment="center", verticalalignment="top", multialignment="left",
                transform = ax.transAxes)

        # Add legend on mid-plane panel
        if fit_type.orientation == "midPlane":
            ax.legend(loc = "lower center")

    fig.tight_layout()
    if filename:
        fig.savefig(filename)

    return fig, axes

def fit_draw_func(rp_fit: fit.ReactionPlaneFit, fit_type: fit.FitType, x: np.ndarray, hist: np.ndarray, ax):
    """ Determine and draw the fit and data on a given axis.

    Args:
        rp_fit: The reaction plane fit object.
        fit_type: The type of the fit component that should be plotted.
        x: x values where the points should be plotted.
        hist: Histogram data for the given fit component.
        ax: matplotlib axis where the information should be plotted.
    Returns:
        None. The current axis is modified.
    """
    # Determine the values of the fit function.
    fit_values = rp_fit.evaluate_fit_component(fit_component = fit_type, x = x)

    # Plot the main values
    plot = ax.plot(x, fit_values, label = "Fit")
    # Plot the fit errors
    errors = rp_fit.fit_result.components[fit_type].errors
    ax.fill_between(x, fit_values - errors, fit_values + errors, facecolor = plot[0].get_color(), alpha = 0.8)
    # Plot the data
    ax.errorbar(x, hist.y, yerr = hist.errors, label = "Data", marker = "o")

def draw_fit(rp_fit: fit.ReactionPlaneFit, data: dict, filename: str):
    """ Main entry point to draw the fit and the data together.

    Args:
        rp_fit: The reaction plane fit object.
        data: Data used to perform the fit.
        filename: Location where the file should be saved.
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which everything has
            been drawn, as well as the axes used for drawing.
    """
    return draw(rp_fit = rp_fit, data = data, filename = filename, draw_func = fit_draw_func, y_label = r"dN/d$\Delta\varphi$")

def residual_draw_func(rp_fit: fit.ReactionPlaneFit, fit_type: fit.FitType, x: np.ndarray, hist: np.ndarray, ax):
    """ Calculate and draw the residual for a given component on a given axis.

    Args:
        rp_fit: The reaction plane fit object.
        fit_type: The type of the fit component that should be plotted.
        x: x values where the points should be plotted.
        hist: Histogram data for the given fit component.
        ax: matplotlib axis where the information should be plotted.
    Returns:
        None. The current axis is modified.
    """
    # Note: Residual = data - fit /fit, not just data-fit
    fit_values = rp_fit.evaluate_fit_component(fit_component = fit_type, x = x)
    residual = (hist.y - fit_values) / fit_values

    # Plot the main values
    ax.plot(x, residual, label = "Residual")

    # Set the y-axis limit to be symmetric
    # Selected the value by looking at the data.
    ax.set_ylim(bottom = -0.1, top = 0.1)

def draw_residual(rp_fit: fit.ReactionPlaneFit, data: dict, filename: str):
    """ Main entry point to draw the residual of the fit.

    Residual is defined as (data-fit)/fit.

    Args:
        rp_fit: The reaction plane fit object.
        data: Data used to perform the fit.
        filename: Location where the file should be saved.
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which everything has
            been drawn, as well as the axes used for drawing.
    """
    return draw(rp_fit = rp_fit, data = data, filename = filename, draw_func = residual_draw_func, y_label = "(data-fit)/fit")

