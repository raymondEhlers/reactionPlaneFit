#!/usr/bin/env python

""" Plotting helper functions related to the Reaction Plane Fit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

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

        # Determine the values of the fit function.
        fit_values = rp_fit.evaluate_fit_component(fit_component = fit_type, x = x)
        # Plot the fit
        plot = ax.plot(x, fit_values)
        # Plot the fit errors
        ax.fill_between(x, fit_values - component_fit_result.errors, fit_values + component_fit_result.errors, facecolor = plot[0].get_color(), alpha = 0.8)
        # Plot the data
        ax.errorbar(x, hist.y, yerr = hist.errors, marker = "o")

    # Add chi2/ndf
    text = r"$\chi^{2}$/NDF = %(chi2).1f/%(ndf)i = %(chi2_over_ndf).3f" % {"chi2": rp_fit.fit_result.minimum_val, "ndf": rp_fit.fit_result.nDOF, "chi2_over_ndf": rp_fit.fit_result.minimum_val / rp_fit.fit_result.nDOF}
    ax.text(0.5, 0.9, text,
            horizontalalignment='center', verticalalignment='center', multialignment="left",
            transform = ax.transAxes)

    fig.tight_layout()
    fig.show()
    fig.savefig("test.png")

def draw_residual(rp_fit):
    #import matplotlib.pyplot as plt
    pass


