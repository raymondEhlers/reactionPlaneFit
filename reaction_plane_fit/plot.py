#!/usr/bin/env python

""" Plotting helper functions related to the Reaction Plane Fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
from typing import Any, Callable, Dict, Tuple

from pachyderm import histogram

from reaction_plane_fit import base
from reaction_plane_fit import fit
from reaction_plane_fit.fit import Data

logger = logging.getLogger(__name__)

# Typing helpers
# This is a tuple of the matplotlib figure and axes. However, we only specify Any here
# because we don't want an explicit package dependency on matplotlib.
Axes = Any
Figure = Any
DrawResult = Tuple[Figure, Axes]

class AnalysisColors:
    """ Exceedingly simple class to store analysis colors. """
    signal = "tab:blue"  # "C0" in the default MPL color cycle
    background = "tab:orange"  # "C1" in the default MPL color cycle
    fit = "tab:green"  # "C2" in the default MPL color cycle
    fit_background = "tab:purple"  # "C4" in the default MPL color cycle

def format_rp_labels(rp_label: str) -> str:
    """ Helper function for formatting the RP label.

    Args:
        rp_label: Reaction plane orientation label to be formatted.
    Returns:
        Properly formatted RP label.
    """
    # Replace "_" with "-"
    return rp_label.replace("_", "-").capitalize()

def draw(rp_fit: fit.ReactionPlaneFit, data: Data, fit_label: str,
         filename: str, y_label: str, draw_func: Callable[..., np.ndarray]) -> DrawResult:
    """ Main RP fit drawing function.

    It takes a callable to do the actual extraction of the relevant data and the plotting. This way, we can
    avoid repeating ourselves for mostly identical plots that include slightly different data.

    Args:
        rp_fit: Reaction plane fit object after fitting.
        data: Data used to perform the fit.
        fit_label: Label for the fit.
        filename: Location where the file should be saved.
        y_label: Label for the y-axis.
        draw_func: Function to take the given data and draw the necessary plots on the given axis.
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which
            everything has been drawn, as well as the axes used for drawing.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})

    # Setup
    full_set_of_components = rp_fit.create_full_set_of_components(input_data = data)
    x = rp_fit.fit_result.x
    n_components = len(full_set_of_components)
    fig, axes = plt.subplots(1, n_components, sharey = True, sharex = True, figsize = (3 * n_components, 6))

    # Shared y axis label
    axes[0].set_ylabel(y_label)

    logger.debug(f"data: {data}")
    for (rp_orientation, component), ax in zip(full_set_of_components.items(), axes):
        # Get the relevant data
        data_for_plotting: Dict[base.FitType, histogram.Histogram1D] = {}
        for fit_type, hist in data.items():
            if fit_type.orientation == rp_orientation:
                data_for_plotting[fit_type] = hist

        # Draw the given data.
        draw_func(data_for_plotting = data_for_plotting, component = component, x = x, ax = ax)

        # Add axis labels
        ax.set_xlabel(r"$\Delta\varphi$")

        # Add fit_type labels
        text = format_rp_labels(rp_orientation)
        # Add in the fit information in the second to last panel
        if ax == axes[-2]:
            text += "\n" + fit_label
        # Add the chi2/ndf for in the last panel.
        if ax == axes[-1]:
            text += "\n" + r"$\chi^{2}$/NDF = %(chi2).1f/%(ndf)i = %(chi2_over_ndf).3f" % {
                "chi2": rp_fit.fit_result.minimum_val,
                "ndf": rp_fit.fit_result.nDOF,
                "chi2_over_ndf": rp_fit.fit_result.minimum_val / rp_fit.fit_result.nDOF
            }
        ax.text(0.5, 0.975, text,
                horizontalalignment="center", verticalalignment="top", multialignment="left",
                transform = ax.transAxes)

    # Add the legend on the inclusive signal axis if it exists.
    # Otherwise, put in on the index 1 axis (which should be mid-pane)
    legend_axis_index = 1
    if len(axes) == 4:
        legend_axis_index = 0
    # Add legend on mid-plane panel
    axes[legend_axis_index].legend(loc = "lower center")

    # Final adjustments
    fig.tight_layout()
    # Adjust spacing between plots
    fig.subplots_adjust(wspace = 0.01)
    # To ensure that this function can be used generally, we only save if a filename is explicitly provided.
    if filename:
        fig.savefig(filename)

    return fig, axes

def fit_draw_func(data_for_plotting: Dict[base.FitType, histogram.Histogram1D], component: fit.FitComponent,
                  x: np.ndarray, ax: Axes) -> None:
    """ Determine and draw the fit and data on a given axis.

    Here, we will draw both the signal and the background dominated data, regardless of what was
    actually used for the fitting.

    Args:
        data_for_plotting: Data to be used for plotting.
        component: RP fit component.
        x: x values where the points should be plotted.
        ax: matplotlib axis where the information should be plotted.
    Returns:
        None. The current axis is modified.
    """
    # Determine the region of the fit.
    # It will be the same for all of the plotting data, so we just take the first one.
    fit_type, hist = next(iter(data_for_plotting.items()))
    # Need to scale the inclusive orientation background down by a factor of 3 because we haven't
    # scaled by number of triggers. Note that this scaling is only approximate, and is just for
    # convenience when plotting.
    scale_factor = 1.
    if fit_type.orientation == "inclusive":
        scale_factor = 1. / 3.

    # Determine the values and the errors of the fit function.
    fit_values = component.evaluate_fit(x = x)
    errors = component.fit_result.errors
    fit_hist = histogram.Histogram1D(bin_edges = hist.bin_edges, y = fit_values, errors_squared = errors ** 2)
    fit_hist *= scale_factor

    # Plot the main values
    plot = ax.plot(x, fit_hist.y, label = "Fit", color = AnalysisColors.fit)
    # Plot the fit errors
    ax.fill_between(
        x, fit_hist.y - fit_hist.errors, fit_hist.y + fit_hist.errors,
        facecolor = plot[0].get_color(), alpha = 0.8, zorder = 2
    )

    # Plot the data
    for fit_type, hist in data_for_plotting.items():
        h_component = hist.copy()
        h_component *= scale_factor
        ax.errorbar(
            x, h_component.y, yerr = h_component.errors, label = f"{fit_type.region.capitalize()} dom. data",
            marker = "o", linestyle = "", fillstyle = "none" if fit_type.region == "background" else "full",
            color = AnalysisColors.signal if fit_type.region == "signal" else AnalysisColors.background,
        )

        # Also plot the background only fit function if the given component fit to the signal region and
        # we have the signal data. We plot it last so that the colors are consistent throughout all axes.
        if isinstance(component, fit.SignalFitComponent) and fit_type.region == "signal":
            # Calculate background function values
            values = component.evaluate_background(x)
            errors = component.calculate_background_function_errors(x)
            fit_hist_component = histogram.Histogram1D(
                bin_edges = hist.bin_edges, y = values, errors_squared = errors ** 2
            )
            fit_hist_component *= scale_factor
            # Plot background values and errors behind everything else
            plot_background = ax.plot(
                fit_hist_component.x, fit_hist_component.y, zorder = 1,
                label = "Bkg. component", color = AnalysisColors.fit_background
            )
            ax.fill_between(
                fit_hist_component.x,
                fit_hist_component.y - fit_hist_component.errors, fit_hist_component.y + fit_hist_component.errors,
                facecolor = plot_background[0].get_color(), alpha = 0.8, zorder = 1,
            )

def draw_fit(rp_fit: fit.ReactionPlaneFit, data: Data, fit_label: str, filename: str) -> DrawResult:
    """ Main entry point to draw the fit and the data together.

    Args:
        rp_fit: The reaction plane fit object.
        data: Data used to perform the fit.
        fit_label: Label for the fit.
        filename: Location where the file should be saved. Set to empty string to avoid printing (for example, if
            you want to handle it externally).
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which
            everything has been drawn, as well as the axes used for drawing.
    """
    return draw(
        rp_fit = rp_fit, data = data, filename = filename, fit_label = fit_label,
        draw_func = fit_draw_func, y_label = r"dN/d$\Delta\varphi$"
    )

def residual_draw_func(data_for_plotting: Dict[base.FitType, histogram.Histogram1D], component: fit.FitComponent,
                       x: np.ndarray, ax: Axes) -> None:
    """ Calculate and draw the residual for a given component on a given axis.

    Args:
        data_for_plotting: Data to be used for plotting.
        component: RP fit component.
        x: x values where the points should be plotted.
        ax: matplotlib axis where the information should be plotted.
    Returns:
        None. The current axis is modified.
    """
    # Determine the hist we are going to use for calculating the residual.
    for fit_type, hist in data_for_plotting.items():
        if fit_type.region == ("signal" if isinstance(component, fit.SignalFitComponent) else "background"):
            break

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
    plot = ax.plot(x, residual.y, label = "Residual", color = AnalysisColors.fit)
    # Plot the fit errors
    ax.fill_between(
        x, residual.y - residual.errors, residual.y + residual.errors,
        facecolor = plot[0].get_color(), alpha = 0.9,
    )

    # Set the y-axis limit to be symmetric
    # Selected the value by looking at the data.
    ax.set_ylim(bottom = -0.25, top = 0.25)

def draw_residual(rp_fit: fit.ReactionPlaneFit, data: Data, fit_label: str, filename: str) -> DrawResult:
    """ Main entry point to draw the residual of the fit.

    Residual is defined as (data-fit)/fit.

    Args:
        rp_fit: The reaction plane fit object.
        data: Data used to perform the fit.
        fit_label: Label for the fit.
        filename: Location where the file should be saved. Set to empty string to avoid printing (for example, if
            you want to handle it externally).
    Returns:
        plt.figure.Figure, np.ndarray[matplotlib.axes._subplots.AxesSubplot]: Matplotlib figure onto which
            everything has been drawn, as well as the axes used for drawing.
    """
    return draw(
        rp_fit = rp_fit, data = data, filename = filename, fit_label = fit_label,
        draw_func = residual_draw_func, y_label = "(data-fit)/fit"
    )

