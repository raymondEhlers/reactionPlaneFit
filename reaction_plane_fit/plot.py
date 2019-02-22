#!/usr/bin/env python

""" Plotting helper functions related to the Reaction Plane Fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import numpy as np
from typing import Any, Callable, Tuple

from pachyderm import histogram

from reaction_plane_fit import base
from reaction_plane_fit import fit
from reaction_plane_fit.fit import Data

# Typing helpers
# This is a tuple of the matplotlib figure and axes. However, we only specify Any here
# because we don't want an explicit package dependency on matplotlib.
Axes = Any
Figure = Any
DrawResult = Tuple[Figure, Axes]

def format_rp_labels(rp_label: str) -> str:
    """ Helper function for formatting the RP label.

    Args:
        rp_label: Reaction plane orientation label to be formatted.
    Returns:
        Properly formatted RP label.
    """
    # Replace "_" with "-"
    return rp_label.replace("_", "-").capitalize()

def draw(rp_fit: fit.ReactionPlaneFit, data: Data, filename: str, y_label: str, draw_func: Callable[..., np.ndarray]) -> DrawResult:
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
        draw_func(fit_type = fit_type, component = component, x = x, hist = hist, ax = ax)

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

    # Add the legend on the inclusive signal axis if it exists.
    # Otherwise, put in on the index 2 axis (which should be mid-pane)
    legend_axis_index = 2
    if len(axes) == 4:
        legend_axis_index = 0
    # Add legend on mid-plane panel
    axes[legend_axis_index].legend(loc = "lower center")

    fig.tight_layout()
    if filename:
        fig.savefig(filename)

    return fig, axes

def fit_draw_func(component: fit.FitComponent, fit_type: base.FitType, x: np.ndarray, hist: np.ndarray, ax: Axes) -> None:
    """ Determine and draw the fit and data on a given axis.

    Args:
        fit_type: The type of the fit component that should be plotted.
        component: RP fit component.
        x: x values where the points should be plotted.
        hist: Histogram data for the given fit component.
        ax: matplotlib axis where the information should be plotted.
    Returns:
        None. The current axis is modified.
    """
    # Determine the values of the fit function.
    fit_values = component.evaluate_fit(x = x)

    # Plot the main values
    plot = ax.plot(x, fit_values, label = "Fit")
    # Plot the fit errors
    errors = component.fit_result.errors
    ax.fill_between(x, fit_values - errors, fit_values + errors, facecolor = plot[0].get_color(), alpha = 0.8, zorder = 2)
    # Plot the data
    ax.errorbar(x, hist.y, yerr = hist.errors, label = "Data", marker = "o", linestyle = "")

    # Also plot the background only function if relevant.
    # We plot it last so that the colors are consistent throughout all axes.
    if fit_type.region == "signal":
        # Calculate background function values
        values = component.evaluate_background(x)
        errors = component.calculate_background_function_errors(x)
        # Plot background values and errors behind everything else
        plot_background = ax.plot(x, values, zorder = 1, label = "Bkg. component")
        ax.fill_between(
            x, values - errors, values + errors,
            facecolor = plot_background[0].get_color(), alpha = 0.8,
            zorder = 1,
        )

def draw_fit(rp_fit: fit.ReactionPlaneFit, data: Data, filename: str) -> DrawResult:
    """ Main entry point to draw the fit and the data together.

    Args:
        rp_fit: The reaction plane fit object.
        data: Data used to perform the fit.
        filename: Location where the file should be saved. Set to empty string to avoid printing (for example, if
            you want to handle it externally).
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which everything has
            been drawn, as well as the axes used for drawing.
    """
    return draw(rp_fit = rp_fit, data = data, filename = filename, draw_func = fit_draw_func, y_label = r"dN/d$\Delta\varphi$")

def residual_draw_func(component: fit.FitComponent, fit_type: base.FitType, x: np.ndarray, hist: np.ndarray, ax: Axes) -> None:
    """ Calculate and draw the residual for a given component on a given axis.

    Args:
        fit_type: The type of the fit component that should be plotted.
        component: RP fit component.
        x: x values where the points should be plotted.
        hist: Histogram data for the given fit component.
        ax: matplotlib axis where the information should be plotted.
    Returns:
        None. The current axis is modified.
    """
    # NOTE: Residual = data - fit / fit, not just data-fit
    # We create a histogram to represent the fit so that we can take advantage
    # of the error propagation in the Histogram1D object.
    fit_hist = histogram.Histogram1D(
        # Bin edges must be the same
        bin_edges = hist.bin_edges,
        y = component.evaluate_fit(x = x),
        errors_squared = component.fit_result.errors ** 2,
    )
    # NOTE: Residual = data - fit / fit, not just data-fit
    residual = (hist - fit_hist) / fit_hist

    # Plot the main values
    plot = ax.plot(x, residual.y, label = "Residual")
    # Plot the fit errors
    ax.fill_between(
        x, residual.y - residual.errors, residual.y + residual.errors,
        facecolor = plot[0].get_color(), alpha = 0.9,
    )

    # Set the y-axis limit to be symmetric
    # Selected the value by looking at the data.
    ax.set_ylim(bottom = -0.1, top = 0.1)

def draw_residual(rp_fit: fit.ReactionPlaneFit, data: Data, filename: str) -> DrawResult:
    """ Main entry point to draw the residual of the fit.

    Residual is defined as (data-fit)/fit.

    Args:
        rp_fit: The reaction plane fit object.
        data: Data used to perform the fit.
        filename: Location where the file should be saved. Set to empty string to avoid printing (for example, if
            you want to handle it externally).
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which everything has
            been drawn, as well as the axes used for drawing.
    """
    return draw(rp_fit = rp_fit, data = data, filename = filename, draw_func = residual_draw_func, y_label = "(data-fit)/fit")

