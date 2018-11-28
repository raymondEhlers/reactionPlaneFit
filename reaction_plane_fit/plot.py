#!/usr/bin/env python

""" Plotting helper functions related to the Reaction Plane Fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import probfit

from reaction_plane_fit import fit

# TODO: Implement these basic plots.

def draw_fit(rp_fit: fit.ReactionPlaneFit, data: dict):
    """ Draw the RP fit with the underlying data.

    Args:
        rp_fit: Reaction plane fit object after fittign.
        data: Data used to perform the fit.
    Returns:
        plt.figure: matplotlib figure onto which everything has been drawn.
    """
    import matplotlib.pyplot as plt

    # Setup
    x = rp_fit.fit_result.x
    n_components = len(rp_fit.components)
    fig, axes = plt.subplots(1, n_components, sharey = True, sharex = True, figsize = (3 * n_components, 6))

    #for (fit_type, component), component_fit_result, ax in zip(rp_fit.components.items(), rp_fit.fit_results.components, axes):
    for (fit_type, component), ax in zip(rp_fit.components.items(), axes):
        # Get the relevant fit and data information
        component_fit_result = rp_fit.fit_result.components[fit_type]
        hist = data[fit_type]

        print(f"component_fit_result args: {component_fit_result.values_at_minimum}")

        # Determine the values of the fit function.
        fit_values = probfit.nputil.vector_apply(component.fit_function, x, *list(component_fit_result.values_at_minimum.values()))
        print(f"fit_values: {fit_values}")
        # Plot the fit
        plot = ax.plot(x, fit_values)
        # Plot the fit errors
        ax.fill_between(x, fit_values - component_fit_result.errors, fit_values + component_fit_result.errors, facecolor = plot[0].get_color(), alpha = 0.8)
        # Plot the data
        ax.errorbar(x, hist.y, yerr = hist.errors, marker = "o")

    fig.tight_layout()
    fig.show()
    #fig.savefig("test.png")

def draw_residual(rp_fit):
    #import matplotlib.pyplot as plt
    pass


