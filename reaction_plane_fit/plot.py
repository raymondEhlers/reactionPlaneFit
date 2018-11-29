#!/usr/bin/env python

""" Plotting helper functions related to the Reaction Plane Fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from re import finditer

from reaction_plane_fit import fit

# TODO: Implement these basic plots.

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

def draw_fit(rp_fit: fit.ReactionPlaneFit, data: dict):
    """ Draw the RP fit with the underlying data.

    Args:
        rp_fit: Reaction plane fit object after fittign.
        data: Data used to perform the fit.
    Returns:
        plt.figure: matplotlib figure onto which everything has been drawn.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})

    # Setup
    x = rp_fit.fit_result.x
    n_components = len(rp_fit.components)
    fig, axes = plt.subplots(1, n_components, sharey = True, sharex = True, figsize = (3 * n_components, 6))

    # Shared y axis label
    axes[0].set_ylabel(r"dN/d$\Delta\varphi$")

    #for (fit_type, component), component_fit_result, ax in zip(rp_fit.components.items(), rp_fit.fit_results.components, axes):
    for (fit_type, component), ax in zip(rp_fit.components.items(), axes):
        # Get the relevant fit and data information
        component_fit_result = rp_fit.fit_result.components[fit_type]
        hist = data[fit_type]

        # Determine the values of the fit function.
        fit_values = rp_fit.evaluate_fit_component(fit_component = fit_type, x = x)
        # Plot the fit
        plot = ax.plot(x, fit_values, label = "Fit")
        # Plot the fit errors
        ax.fill_between(x, fit_values - component_fit_result.errors, fit_values + component_fit_result.errors, facecolor = plot[0].get_color(), alpha = 0.8)
        # Plot the data
        ax.errorbar(x, hist.y, yerr = hist.errors, label = "Data", marker = "o")
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
    fig.savefig("example.png")

    return fig, axes

def draw_residual(rp_fit):
    #import matplotlib.pyplot as plt
    pass

